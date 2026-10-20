"""Short-horizon, motor-only reach planning in isolated prediction data.

This is model-based control, NOT PPO. No prediction overwrites live state.
The real rollout retains the original substep audits and independent gate.
"""
from concurrent.futures import ThreadPoolExecutor
import time
import mujoco
import numpy as np
from prediction_stepper import PredictionStepper
from grasp_forecast import forecast_grasp,check_prediction_deadline

class PredictiveReach:
    def __init__(self,r,horizon=.2,workers=4,deadline=None):
        if deadline is not None and not np.isfinite(deadline):raise ValueError('Invalid prediction deadline')
        self.deadline=deadline
        self.r=r;self.horizon=r.config.get('predictionHorizon',horizon);self.position_weight=r.config.get('predictionPositionWeight',1.);self.velocity_seconds=r.config.get('predictionVelocitySeconds',0.)
        if not .2<=self.horizon<=.6 or not 1<=self.position_weight<=4 or not 0<=self.velocity_seconds<=.3:raise ValueError('Predictive cost settings outside bounded prototype range')
        self.grip_offset=np.asarray(r.config.get('gripOffset',[-.001,-.05,.003]),dtype=float)
        if self.grip_offset.shape!=(3,) or not np.isfinite(self.grip_offset).all() or np.max(np.abs(self.grip_offset))>.08:raise ValueError('Invalid local grip offset')
        self.cloth_node=None
        self.pool=ThreadPoolExecutor(max_workers=workers);self.ctrl=None;self.updates=0
        self.lower=r.model.jnt_range[r.joint_ids,0]+.04;self.upper=r.model.jnt_range[r.joint_ids,1]-.04
        self.jaw=.32;self.last_time=-1;self.last_plan=None;self.total_seconds=0
        self.stage='reach';self.transport_goal=None
    def features(self,d):
        r=self.r;R=d.xmat[r.head].reshape(3,3);p=d.xpos[r.head]+R@(np.array([-.016,0.,-.075])+self.grip_offset)
        feet=d.geom_xpos[r.feet].reshape(-1)
        com=d.subtree_com[r.trunk,:2]
        features=np.r_[p*self.position_weight,R.reshape(-1)*.02,feet*2,com]
        if self.velocity_seconds:
            # Penalize terminal grip-point speed as a distance over this time
            # scale. Reaching a safe position at high speed can leave no safe
            # continuation at the next receding-horizon update.
            jp=np.zeros((3,r.model.nv));jr=np.zeros_like(jp);mujoco.mj_jac(r.model,d,jp,jr,p,r.head)
            features=np.r_[features,(jp@d.qvel)*self.velocity_seconds]
        return features
    def set_cloth_node(self,node):
        r=self.r
        if r.config.get('kind')!='sock' or not r.config.get('raisedCloth') or r.flex<0:
            raise ValueError('A tracked material node requires the raised-cloth task')
        if isinstance(node,bool) or not isinstance(node,int) or not 0<=node<int(r.model.flex_vertnum[r.flex]):
            raise ValueError('Invalid tracked cloth node')
        if self.cloth_node is not None and node!=self.cloth_node:
            raise ValueError('A prediction session must retain its material node')
        self.cloth_node=node
    def goal(self,point):
        if self.stage=='transport':return self.transport_goal.copy()
        r=self.r;s=r.snapshot();w,x,y,z=s['payload'][3:];yaw=np.arctan2(-2*(x*y-w*z),1-2*(x*x+z*z))+.036
        roll=.076;pitch=-.098;cr,sr=np.cos(roll),np.sin(roll);cp,sp=np.cos(pitch),np.sin(pitch);cy,sy=np.cos(yaw),np.sin(yaw)
        R=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
        target=np.array(s['payload'][:3])
        if r.config.get('raisedCloth'):
            if self.cloth_node is None:raise ValueError('Missing tracked cloth node')
            target=np.array(s['clothPoints'][self.cloth_node])
        if r.config.get('axisGrasp') and r.config.get('kind')=='pen':
            axis=np.array([2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)])
            half=max(0,r.config['length']/2-.02)
            target+=np.clip(np.dot(point-target,axis),-half,half)*axis
        rest=self.reference[12:].copy()
        if self.velocity_seconds:rest[-3:]=0
        return np.r_[target*self.position_weight,R.reshape(-1)*.02,rest]
    def predict(self,ctrl):
        check_prediction_deadline(self.deadline)
        r=self.r;q=mujoco.MjData(r.model);mujoco.mj_copyData(q,r.model,r.data);q.ctrl[:14]=ctrl;q.ctrl[14]=self.jaw
        stepper=PredictionStepper(r.stepper,q)
        try:
            a=stepper.step(round(self.horizon/r.model.opt.timestep))
            check_prediction_deadline(self.deadline)
            height=min(float((v@q.geom_xmat[g].reshape(3,3).T+q.geom_xpos[g])[:,2].min()) for g,v in r.beak_meshes)
            lower=upper=0.
            if self.stage=='transport':
                force=np.zeros(6)
                for contact_index,contact in enumerate(q.contact):
                    first=(r.payload_geom>=0 and contact.geom[0]==r.payload_geom) or (r.flex>=0 and contact.flex[0]==r.flex)
                    second=(r.payload_geom>=0 and contact.geom[1]==r.payload_geom) or (r.flex>=0 and contact.flex[1]==r.flex)
                    if first==second or contact.dist>0 or contact.efc_address<0:continue
                    other=contact.geom[1] if first else contact.geom[0]
                    if other<0:continue
                    mujoco.mj_contactForce(r.model,q,contact_index,force)
                    if r.stepper.classes[other]&4:lower+=float(force[0])
                    elif r.stepper.classes[other]&8:upper+=float(force[0])
            lost_grasp=self.stage=='transport' and min(lower,upper)<=.002
            unsafe=bool(a[10] or not a[12] or a[13] or a[14] or a[18] or lost_grasp or max(a[7],a[8])>=.0005 or a[11]>=.002)
            return dict(features=self.features(q),root=q.qpos[:7].tolist(),headQuat=q.xquat[r.head].tolist(),height=height,unsafe=unsafe,physical=dict(nonFootSupport=bool(a[10]),finite=bool(a[12]),solverReset=bool(a[13]),maxExternal=float(a[14]),warnings=int(a[18]),lostGrasp=lost_grasp,finalLowerForce=lower,finalUpperForce=upper,maxPenetration=float(a[7]),maxRobotPenetration=float(a[8]),maxJointLimitExcess=float(a[11])))
        finally:stepper.close()
    def update(self,stage='reach'):
        check_prediction_deadline(getattr(self,'deadline',None))
        r=self.r
        if stage not in ['reach','transport']:raise ValueError('Unknown predictive stage')
        if stage!=self.stage:
            if stage!='transport':raise ValueError('Create a new planner for a new reach attempt')
            if self.ctrl is None or not hasattr(self,'reference'):raise ValueError('Transport requires a preceding reach observation')
            lift=r.config.get('transportLiftHeight',.07)
            if not .05<=lift<=.18:raise ValueError('Transport height outside bounded prototype range')
            self.stage=stage;self.jaw=0;self.ctrl=np.clip(r.data.ctrl[:14].copy(),self.lower,self.upper)
            self.transport_goal=self.features(r.data);self.transport_goal[2]+=lift*self.position_weight
            # Preserve the supported feet/COM reference from reaching while
            # holding the measured grasp orientation. No body state is set.
            self.transport_goal[12:20]=self.reference[12:20]
            if self.velocity_seconds:self.transport_goal[-3:]=0
            self.last_time=-1
        if self.last_plan is not None and r.data.time-self.last_time<.199:return self.last_plan
        started=time.monotonic();self.last_time=r.data.time
        if self.ctrl is None:self.ctrl=np.clip(r.data.ctrl[:14].copy(),self.lower,self.upper);self.reference=self.features(r.data)
        n=mujoco.mj_stateSize(r.model,mujoco.mjtState.mjSTATE_INTEGRATION);before=np.zeros(n);after=np.zeros(n)
        mujoco.mj_getState(r.model,r.data,before,mujoco.mjtState.mjSTATE_INTEGRATION)
        baseline=self.predict(self.ctrl);goal=self.goal(baseline['features'][:3]/self.position_weight);epsilon=.025;variants=[]
        for j in range(14):
            for sign in [-1,1]:
                c=self.ctrl.copy();c[j]+=sign*epsilon;variants.append(np.clip(c,self.lower,self.upper))
        predictions=list(self.pool.map(self.predict,variants));columns=[];floor=[]
        for j in range(14):
            minus,plus=predictions[2*j:2*j+2];den=variants[2*j+1][j]-variants[2*j][j]
            columns.append((plus['features']-minus['features'])/max(den,1e-8));floor.append((plus['height']-minus['height'])/max(den,1e-8))
        J=np.array(columns).T;error=goal-baseline['features'];matrix=J.T@J+np.eye(14)*.003**2;rhs=J.T@error
        delta=np.linalg.solve(matrix,rhs);normal=np.array(floor);required=.0015-baseline['height']
        if normal@delta<required:
            response=np.linalg.solve(matrix,normal);den=normal@response
            if den>1e-10:delta+=response*(required-normal@delta)/den
        delta*=min(1,.06/max(float(np.max(np.abs(delta))),1e-12))
        candidates=[self.ctrl]+[np.clip(self.ctrl+scale*delta,self.lower,self.upper) for scale in [1,.5,.25]]
        evaluations=[baseline,*self.pool.map(self.predict,candidates[1:])]
        # These 28 motor-only forecasts have already been computed. They are
        # useful safe fallbacks when the coupled linear proposal crosses a
        # contact discontinuity; retaining them adds no extra simulation work.
        candidates.extend(variants);evaluations.extend(predictions)
        def cost(p):return float(np.sum((p['features']-goal)**2))+100*max(0,.0015-p['height'])**2+(1 if p['unsafe'] else 0)
        safe=[i for i,p in enumerate(evaluations) if not p['unsafe']]
        blocked=not safe
        best=min(safe,key=lambda i:cost(evaluations[i])) if safe else 0
        closure=None
        if self.stage=='reach' and not blocked and r.config.get('graspForecast') and np.linalg.norm(evaluations[best]['features'][:3]-goal[:3])/self.position_weight<.006:
            shortlist=sorted(safe,key=lambda i:cost(evaluations[i]))[:8]
            closures=list(self.pool.map(lambda i:forecast_grasp(r,candidates[i],deadline=self.deadline),shortlist))
            graspable=[i for i,c in zip(shortlist,closures) if c['graspable']]
            if graspable:best=min(graspable,key=lambda i:cost(evaluations[i]))
            closure=closures[shortlist.index(best)]
        if not blocked:self.ctrl=candidates[best]
        mujoco.mj_getState(r.model,r.data,after,mujoco.mjtState.mjSTATE_INTEGRATION);np.testing.assert_array_equal(before,after)
        self.updates+=1;elapsed=time.monotonic()-started;self.total_seconds+=elapsed
        self.last_plan=dict(stage=self.stage,ctrl=self.ctrl.tolist(),blocked=blocked,graspReady=bool(closure and closure['graspable']),closure=closure,audit=dict(sourceStateUnchanged=True,baselineCost=cost(baseline),chosenCost=cost(evaluations[best]),candidate=best,
            safeCandidates=len(safe),predictedPhysical=evaluations[best]['physical'],predictedUnsafe=evaluations[best]['unsafe'],predictedClearance=evaluations[best]['height'],predictedRoot=evaluations[best].get('root'),predictedHeadQuat=evaluations[best].get('headQuat'),update=self.updates,wallSeconds=elapsed))
        return self.last_plan
    def close(self):self.pool.shutdown(wait=True)
