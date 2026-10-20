"""Observation-dependent reach corrections under the unchanged physics gate.

Training resets may restore cached, physically executed prefixes. Full
deployment validation must start from the normal ground reset, never a cache.
"""
import copy
import gzip
import json
import os
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from native_household import NativeHousehold
from run_native_household import ControllerBridge

class HouseholdResidualEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,prefix_dir,trace_dir=None,fixed_prefix=None):
        super().__init__()
        self.observation_space=gym.spaces.Box(-5,5,(93,),np.float32)
        self.action_space=gym.spaces.Box(-1,1,(14,),np.float32)
        self.prefixes=[json.loads(p.read_text()) for p in sorted(Path(prefix_dir).glob('prefix-*.json'))]
        if not self.prefixes:raise ValueError('No audited training prefixes')
        self.bridge=ControllerBridge();self.r=None;self.xml=None;self.fixed_prefix=fixed_prefix
        self.contract=self.bridge.request(op='residual-contract')
        if any(p.get('contract')!=self.contract for p in self.prefixes):
            self.bridge.close();raise ValueError('Training prefixes use a different residual contract; regenerate them')
        self.trace_dir=Path(trace_dir) if trace_dir else None;self.episode=0;self.saved=0

    def _observation(self):
        return np.array(self.bridge.request(op='observe',snapshot=self.r.snapshot(),context=self.r.controller_context(False)),np.float32)

    def _distance(self):
        s=self.r.snapshot();R=self.r.data.xmat[self.r.head].reshape(3,3)
        point=np.array(s['graspPoint'])+R@np.array(self.config.get('gripOffset',[-.001,-.05,.003]))
        return float(np.linalg.norm(np.array(s['payload'][:3])-point))

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);self.episode+=1
        index=(options or {}).get('prefix',self.fixed_prefix)
        self.prefix_index=int(self.np_random.integers(len(self.prefixes))) if index is None else int(index)
        prefix=self.prefixes[self.prefix_index];self.config=copy.deepcopy(prefix['config'])
        if self.xml!=prefix['modelXml']:
            if self.r:self.r.close()
            self.r=NativeHousehold(prefix['modelXml'],self.config);self.xml=prefix['modelXml']
        self.r.reset(self.config)
        # Episode reset only. Neither step() nor the shared controller restores
        # a pose or writes any applied force input.
        mujoco.mj_setState(self.r.model,self.r.data,np.array(prefix['physicsState']),mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.r.model,self.r.data)
        self.r.last_action=np.array(prefix['lastAction'],np.float32)
        self.r.summary=copy.deepcopy(prefix['summary']);self.r.contact=copy.deepcopy(prefix['contact'])
        self.rows=copy.deepcopy(prefix['rows']);self.phase='reach';self.ticks=0
        self.bridge.request(op='restore-prefix',state=prefix['controllerState'])
        self.previous_distance=self._distance();self.done=False
        return self._observation(),dict(trainingPrefixReset=True,prefix=self.prefix_index)

    def step(self,action):
        if self.done:raise RuntimeError('Reset after a terminal episode')
        values=np.asarray(action,dtype=np.float64)
        if values.shape!=(14,) or not np.all(np.isfinite(values)):raise ValueError('Expected 14 finite actions')
        action=np.clip(values,-1,1).tolist();unsafe=False
        for _ in range(5):
            result=self.bridge.request(op='step',snapshot=self.r.snapshot(),meta=self.r.metadata(),context=self.r.controller_context(True),summary=self.r.summary,residualAction=action)
            row=self.r.step(result['ctrl']);row['phase']=self.phase=result['phase'];row['reachServo']=result.get('reachServo');self.rows.append(row);self.ticks+=1
            s=self.r.summary
            unsafe=bool(s['nonFootSupport'] or not s['finite'] or s['solverReset'] or s['maxExternal'] or any(s['warningCounts'].values()) or max(s['maxPenetration'],s['maxRobotPenetration'])>=.0005 or s['maxJointLimitExcess']>=.002)
            if unsafe or self.phase in ['done','failed']:break
        row=self.rows[-1];c=row['contact'];distance=self._distance()
        paired=c['lowerForce']>.002 and c['upperForce']>.002
        lifted=paired and row['shape']['minZ']>.025 and not c['nonMouthSupport']
        reward=.1*np.exp(-distance/.012)+20*(self.previous_distance-distance)+.3*paired+2*lifted-.001*float(np.mean(values**2))
        self.previous_distance=distance
        terminated=unsafe or self.phase in ['done','failed'];truncated=not terminated and self.ticks>=1250
        self.done=bool(terminated or truncated)
        info=dict(trainingPrefixReset=True,prefix=self.prefix_index,phase=self.phase,unsafe=unsafe,distance=distance)
        if self.done:
            if unsafe:self.phase='failed';reward-=100
            elif self.phase=='failed' or truncated:reward-=10
            events=self.bridge.request(op='events')
            evidence=dict(config=self.config,phase=self.phase,rows=self.rows,summary=self.r.summary,modelAudit=self.r.model_audit,modelXml=self.xml,trainingPrefixReset=True,**events)
            acceptance=self.bridge.request(op='evaluate',evidence={key:evidence[key] for key in ['phase','rows','summary','modelAudit']})
            evidence['acceptance']=acceptance
            reward+=100*acceptance['accepted'];info['acceptance']=acceptance;info['summary']=copy.deepcopy(self.r.summary)
            if self.trace_dir and (self.saved<20 or acceptance['accepted'] and self.saved<40):
                self.trace_dir.mkdir(parents=True,exist_ok=True);path=self.trace_dir/f'worker-{os.getpid()}-episode-{self.episode}.json.gz'
                with gzip.open(path,'wt') as f:json.dump(evidence,f)
                info['trace']=str(path);self.saved+=1
        return self._observation(),float(reward),bool(terminated),bool(truncated),info

    def close(self):
        if self.r:self.r.close();self.r=None
        self.bridge.close()
