"""Training resets are explicit, and invalid actions cannot touch physics."""
import argparse
import json
import numpy as np
from household_residual_env import HouseholdResidualEnv

p=argparse.ArgumentParser();p.add_argument('--prefixes',required=True);args=p.parse_args()
env=HouseholdResidualEnv(args.prefixes,fixed_prefix=0)
try:
    obs,info=env.reset(seed=1)
    assert info['trainingPrefixReset'] and env.observation_space.contains(obs)
    before=env.r.data.qpos.copy();time=env.r.data.time;ctrl=env.r.data.ctrl.copy()
    for bad in [np.zeros(13),np.r_[np.nan,np.zeros(13)]]:
        try:env.step(bad)
        except ValueError:pass
        else:raise AssertionError('Invalid action accepted')
        np.testing.assert_array_equal(before,env.r.data.qpos);np.testing.assert_array_equal(ctrl,env.r.data.ctrl);assert time==env.r.data.time
    one=env.step(np.zeros(14));position=env.r.data.qpos.copy()
    env.reset(seed=1);two=env.step(np.zeros(14))
    np.testing.assert_allclose(position,env.r.data.qpos,atol=1e-12,rtol=0)
    np.testing.assert_allclose(one[0],two[0],atol=1e-9,rtol=0)
    assert env.observation_space.contains(two[0])
    while not (two[2] or two[3]):two=env.step(np.zeros(14))
    print(json.dumps(dict(test='zero-residual training-prefix baseline',**two[4])))
    assert not env.r.summary['maxExternal'];assert not env.r.summary['solverReset']
    # Establish a physically positive curriculum before dispatching PPO.
    assert two[4]['acceptance']['accepted'],'Zero residual nominal curriculum must pass before training'
    print('PASS: explicit reproducible prefix reset, observation contract, atomic actions and positive zero-residual curriculum')
finally:env.close()
