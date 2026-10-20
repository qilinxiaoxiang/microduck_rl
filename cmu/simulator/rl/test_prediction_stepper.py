"""Independent clone buffers, physical evolution and no repeated C++ builds."""
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import mujoco
import numpy as np
from native_household import NativeHousehold
from prediction_stepper import PredictionStepper
from probe_motor_response import state

p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True);args=p.parse_args()
record=json.loads(args.prefix.read_text())
r=NativeHousehold(record['modelXml'],record['config'])
try:
    before=state(r.model,r.data)
    def run(change):
        q=mujoco.MjData(r.model);mujoco.mj_copyData(q,r.model,r.data);q.ctrl[5]+=change
        predictor=PredictionStepper(r.stepper,q);out=predictor.step(320);predictor.close()
        return out,q.qpos.copy()
    with patch('subprocess.run',side_effect=AssertionError('A prediction must not compile again')):
        with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,[0,0,.1]))
    np.testing.assert_array_equal(results[0][0],results[1][0]);np.testing.assert_array_equal(results[0][1],results[1][1])
    assert np.max(np.abs(results[2][1]-results[0][1]))>1e-8
    np.testing.assert_array_equal(before,state(r.model,r.data));assert all(x[0][12] for x in results)
    print('PASS: independent concurrent prediction data, no recompilation, and unchanged source integration state')
finally:r.close()
