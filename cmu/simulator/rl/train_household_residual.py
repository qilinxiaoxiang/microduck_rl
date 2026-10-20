"""PPO reach residuals. Cached prefixes are training-only, never deployment proof."""
import argparse
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from household_residual_env import HouseholdResidualEnv
from run_native_household import source_hashes,ROOT,ControllerBridge

def actor(model,path):
    modules=[*model.policy.mlp_extractor.policy_net,model.policy.action_net];layers=[]
    for i,module in enumerate(modules):
        if isinstance(module,torch.nn.Linear):
            layers.append(dict(weight=module.weight.detach().cpu().tolist(),bias=module.bias.detach().cpu().tolist(),activation='tanh' if i+1<len(modules) and isinstance(modules[i+1],torch.nn.Tanh) else 'linear'))
    path.write_text(json.dumps(dict(**model.household_contract,layers=layers,scope='reach and clamp residual motor corrections; existing walking and lift remain frozen')))

def make_env(prefixes,traces):return Monitor(HouseholdResidualEnv(prefixes,trace_dir=traces))

class Evidence(BaseCallback):
    def __init__(self,out,seconds):super().__init__();self.out=out;self.seconds=seconds;self.started=time.monotonic();self.episodes=0;self.successes=0
    def _on_step(self):
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            if not done:continue
            self.episodes+=1;self.successes+=bool(info.get('acceptance',{}).get('accepted'))
            row={k:v for k,v in info.items() if k!='terminal_observation'}
            row.update(step=self.num_timesteps,wallSeconds=time.monotonic()-self.started)
            with (self.out/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return time.monotonic()-self.started<self.seconds
    def _on_rollout_end(self):
        self.model.save(self.out/'checkpoint');actor(self.model,self.out/'checkpoint-actor.json')
        print(json.dumps(dict(step=self.num_timesteps,episodes=self.episodes,trainingPrefixPasses=self.successes,seconds=round(time.monotonic()-self.started))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefixes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=8192);p.add_argument('--envs',type=int,default=8);p.add_argument('--seconds',type=int,default=900);p.add_argument('--seed',type=int,default=92651);p.add_argument('--resume',type=Path)
    args=p.parse_args()
    if not 1<=args.envs<=32 or not 1<=args.seconds<=3600 or not 1<=args.steps<=1000000:raise ValueError('Training must be bounded')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    bridge=ControllerBridge()
    try:contract=bridge.request(op='residual-contract')
    finally:bridge.close()
    hashes=source_hashes()
    for name in ['rl/household_residual_env.py','rl/train_household_residual.py','rl/prepare_residual_prefixes.py']:
        hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    manifest=dict(contract=contract,algorithm='Stable-Baselines3 PPO',seed=args.seed,steps=args.steps,envs=args.envs,wallLimitSeconds=args.seconds,decisionHz=10,physicsHz=16000,
                  scope='training-prefix curriculum only, not full start or browser acceptance',sources=hashes,
                  packages={p:importlib.metadata.version(p) for p in ['mujoco','numpy','torch','gymnasium','stable-baselines3']},
                  prefixes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.prefixes.glob('prefix-*.json')})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    env=SubprocVecEnv([partial(make_env,args.prefixes,args.output/'traces') for _ in range(args.envs)],start_method='spawn');env.seed(args.seed)
    try:
        if args.resume:
            model=PPO.load(args.resume,env=env,device='cpu')
            if getattr(model,'household_contract',None)!=contract:raise ValueError('Cannot resume a different observation/action contract')
        else:
            model=PPO('MlpPolicy',env,n_steps=128,batch_size=128,n_epochs=5,learning_rate=3e-4,gamma=.995,gae_lambda=.95,ent_coef=.001,
                      policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),log_std_init=-1.5),seed=args.seed,device='cpu',verbose=0)
            # Zero deterministic residual is the existing controller baseline.
            torch.nn.init.zeros_(model.policy.action_net.weight);torch.nn.init.zeros_(model.policy.action_net.bias)
            model.household_contract=contract
        initial={k:v.detach().clone() for k,v in model.policy.state_dict().items()};model.save(args.output/'initial');actor(model,args.output/'initial-actor.json')
        callback=Evidence(args.output,args.seconds);model.learn(total_timesteps=args.steps,callback=callback,reset_num_timesteps=not bool(args.resume))
        model.save(args.output/'final');actor(model,args.output/'final-actor.json')
        delta=sum(float(((v-initial[k])**2).sum()) for k,v in model.policy.state_dict().items())**.5
        (args.output/'completed.json').write_text(json.dumps(dict(steps=model.num_timesteps,optimizerUpdates=model._n_updates,parameterDeltaL2=delta,trainingPrefixPasses=callback.successes,episodes=callback.episodes,seconds=time.monotonic()-callback.started)))
    finally:env.close()
