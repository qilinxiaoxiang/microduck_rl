"""Forecast the actual rate-limited jaw closure in isolated physics data."""
import mujoco
import numpy as np
import time
from prediction_stepper import PredictionStepper

def check_prediction_deadline(deadline):
    if deadline is not None and (isinstance(deadline,bool) or not isinstance(deadline,(int,float)) or not np.isfinite(deadline)):
        raise ValueError('Invalid prediction deadline')
    if deadline is not None and time.monotonic()>=deadline:
        raise TimeoutError('Predictive-control deadline reached inside a physical forecast')

def forecast_grasp(r,ctrl,*,deadline=None):
    check_prediction_deadline(deadline)
    delay=r.config.get('graspForecastDelay',0.)
    if not isinstance(delay,(int,float)) or not np.isfinite(delay) or not 0<=delay<=.4:raise ValueError('Closure delay must be finite and between zero and 0.4 seconds')
    delay_ticks=round(delay/.02)
    values=np.asarray(ctrl,dtype=float)
    if values.shape!=(14,) or not np.isfinite(values).all():raise ValueError('Expected 14 finite motor targets')
    spec=mujoco.mjtState.mjSTATE_INTEGRATION;n=mujoco.mj_stateSize(r.model,spec)
    before=np.zeros(n);after=np.zeros(n);mujoco.mj_getState(r.model,r.data,before,spec)
    d=mujoco.MjData(r.model);mujoco.mj_copyData(d,r.model,r.data);d.ctrl[:14]=values
    stepper=PredictionStepper(r.stepper,d);stable=0.;unsafe=False;maximum=0.;max_robot=0.;last=None
    try:
        for tick in range(50+delay_ticks):
            check_prediction_deadline(deadline)
            # Same 0.6 rad/s target slew as JawActionFilter at the real 50 Hz
            # control loop. No instantaneous closing target in this forecast.
            if tick>=delay_ticks:d.ctrl[14]=max(0,float(d.ctrl[14])-.6*.02)
            a=stepper.step(round(.02/r.model.opt.timestep));maximum=max(maximum,float(a[7]));max_robot=max(max_robot,float(a[8]))
            check_prediction_deadline(deadline)
            unsafe=bool(a[10] or not a[12] or a[13] or a[14] or a[18] or max(a[7],a[8])>=.0005 or a[11]>=.002)
            paired=tick>=delay_ticks and a[1]>.002 and a[2]>.002;stable=stable+.02 if paired else 0.
            last=dict(lowerForce=float(a[1]),upperForce=float(a[2]),floorForce=float(a[4]),nonFootSupport=bool(a[10]))
            if unsafe:break
        mujoco.mj_getState(r.model,r.data,after,spec);np.testing.assert_array_equal(before,after)
        return dict(graspable=not unsafe and stable>=.2,pairedSeconds=stable,unsafe=unsafe,maxPenetration=maximum,maxRobotPenetration=max_robot,
                    finalContact=last,sourceStateUnchanged=True,openDelaySeconds=delay_ticks*.02,jawTarget=float(d.ctrl[14]),jaw=float(d.qpos[r.jaw]))
    finally:stepper.close()
