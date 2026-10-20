"""Small supervised teacher-to-policy pilot, not reinforcement learning.

Only whole-trial accepted demonstrations are training data. One explicitly
named trajectory is excluded from optimization. Offline label error is not
closed-loop or held-out pickup acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--trials',type=Path,required=True);p.add_argument('--holdout',default='case-007');p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=2500);a=p.parse_args()
if not 1<=a.epochs<=5000:raise ValueError('Bounded supervised pilot required')
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(92652)
train=[];validation=[];manifest=[]
for path in sorted(a.trials.glob('case-*/trial-000.json')):
    d=json.loads(path.read_text());group=path.parent.name
    accepted=d.get('acceptance',{}).get('accepted',False) and not d.get('incomplete') and not d.get('failure')
    samples=[s for s in d.get('teacherSamples',[]) if not s['blocked'] and s['audit']['sourceStateUnchanged'] and not s['audit']['predictedUnsafe']]
    manifest.append(dict(group=group,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),accepted=accepted,samples=len(samples),split='excluded-failure' if not accepted else 'validation' if group==a.holdout else 'train'))
    if not accepted:continue
    rows=[(s['observation'],np.clip((np.array(s['targets'])-s['beforeCtrl'])/.06,-1,1),1/len(samples)) for s in samples]
    (validation if group==a.holdout else train).extend(rows)
if not train or not validation:raise ValueError('Separate accepted training and validation trajectories required')
x=np.array([s[0] for s in train]);mean=x.mean(0);scale=np.maximum(x.std(0),.05)
def tensors(rows):return torch.tensor(np.clip((np.array([r[0] for r in rows])-mean)/scale,-10,10),dtype=torch.float32),torch.tensor(np.array([r[1] for r in rows]),dtype=torch.float32)
x,y=tensors(train);vx,vy=tensors(validation);weights=torch.tensor([r[2] for r in train]);weights/=weights.sum()
model=torch.nn.Sequential(torch.nn.Linear(93,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,14))
optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);started=time.monotonic();history=[]
for epoch in range(a.epochs):
    prediction=model(x);loss=(((prediction-y)**2).mean(1)*weights).sum()
    optimizer.zero_grad();loss.backward();optimizer.step()
    if epoch%100==0 or epoch+1==a.epochs:
        with torch.no_grad():error=((model(vx).clamp(-1,1)-vy)**2).mean().sqrt().item()
        history.append(dict(epoch=epoch+1,trainMSE=loss.item(),validationActionRMSE=error))
layers=[]
for layer in model:
    if isinstance(layer,torch.nn.Linear):layers.append(dict(weight=layer.weight.detach().tolist(),bias=layer.bias.detach().tolist(),activation='linear'))
    else:layers[-1]['activation']='tanh'
actor=dict(version=1,type='household-teacher-motor-delta',observationSize=93,actionSize=14,decisionHz=5,deltaScale=.06,mean=mean.tolist(),scale=scale.tolist(),layers=layers,
           scope='Supervised imitation pilot, not PPO; no physical safety predictor at inference')
(a.output/'actor.json').write_text(json.dumps(actor));torch.save(model.state_dict(),a.output/'weights.pt')
result=dict(algorithm='supervised behavior cloning',seed=92652,epochs=a.epochs,trainSamples=len(train),validationSamples=len(validation),seconds=time.monotonic()-started,sources=manifest,history=history)
(a.output/'training.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['sources','history']}));print(json.dumps(history[-1]))
