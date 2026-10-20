"""Reward-guided controller parameter search, not PPO or a neural RL policy.

Cross-entropy search updates a bounded Gaussian from measured physical
outcomes. No candidate can edit mass, friction, torque, geometry, or acceptance.
The reused development starts must not be called held-out evaluations.
"""
import math
import random
import statistics

BOUNDS=[('crouch',.04,.22),('biteDistance',.055,.09),('neckTarget',-1.53,-1.1),
        ('worldAngle',-.18,.04),('gripX',-.010,.010),('gripZ',0.,.012),
        ('servoRoll',-.05,.18),('servoPitch',-.22,.04)]
INITIAL=[.14,.0787,-1.48,0.,-.001,.003,.076,-.098]


def decode(values, base):
    if len(values)!=len(BOUNDS) or not all(math.isfinite(x) for x in values):
        raise ValueError('Invalid parameter candidate')
    params={name:lo+max(0,min(1,x))*(hi-lo) for x,(name,lo,hi) in zip(values,BOUNDS)}
    grip=[params.pop('gripX'),-.05,params.pop('gripZ')]
    return {**base,**params,'gripOffset':grip}


def physical_score(record):
    if record.get('incomplete') or record.get('failure'):return -10000.
    s=record['summary'];a=record['acceptance'];rows=record['rows']
    if not s['finite'] or s['solverReset'] or s['maxExternal'] or any(s.get('warningCounts',{}).values()):return -10000.
    if s['nonFootSupport'] or max(s['maxPenetration'],s['maxRobotPenetration'])>=.0005 or s['maxJointLimitExcess']>=.002:
        return -1000.-min(100.,s.get('nonFootSeconds',0)*10)-min(100.,max(s['maxPenetration'],s['maxRobotPenetration'])*100000)
    relevant=[r for r in rows if r['phase'] not in ['manual','align','prestand','settle']]
    if not relevant:return -100.
    def seconds(predicate):return .02*sum(bool(predicate(r)) for r in relevant)
    lower=seconds(lambda r:r['contact']['lowerForce']>.002)
    upper=seconds(lambda r:r['contact']['upperForce']>.002)
    opposed=lambda r:r['contact']['lowerForce']>.002 and r['contact']['upperForce']>.002
    paired=seconds(opposed)
    lifted=seconds(lambda r:opposed(r) and not r['contact']['nonMouthSupport'] and r['shape']['minZ']>.025)
    errors=[math.sqrt(sum(x*x for x in r['reachServo']['error'])) for r in relevant if r.get('reachServo') and r['phase']=='reach']
    proximity=5*math.exp(-min(errors)/.005) if errors else 0
    return -20+proximity+10*min(lower,.5)+10*min(upper,.5)+40*min(paired,1)+100*min(lifted,1)+500*bool(a['accepted'])


class CrossEntropySearch:
    def __init__(self,seed=92605):
        self.rng=random.Random(seed)
        self.mean=[(x-lo)/(hi-lo) for x,(_,lo,hi) in zip(INITIAL,BOUNDS)]
        self.std=[.18]*len(BOUNDS);self.best=None

    def ask(self,count=16):
        if not 4<=count<=32:raise ValueError('Population must be 4–32')
        values=[[max(0.,min(1.,self.rng.gauss(mean,std))) for mean,std in zip(self.mean,self.std)] for _ in range(count)]
        values[0]=self.best['values'].copy() if self.best else self.mean.copy()
        return values

    def tell(self,evaluations):
        ranked=sorted(evaluations,key=lambda e:e['score'],reverse=True)
        if self.best is None or ranked[0]['score']>self.best['score']:self.best=ranked[0].copy()
        elites=ranked[:max(2,len(ranked)//4)]
        for j in range(len(self.mean)):
            xs=[e['values'][j] for e in elites]
            self.mean[j]=.5*self.mean[j]+.5*statistics.mean(xs)
            self.std[j]=max(.05,.5*self.std[j]+.5*statistics.pstdev(xs))
        return self.best
