"""Reuse the audited native stepping function for isolated prediction data.

Each predictor owns its output buffer and MuJoCo data. The compiled function,
read-only model and geometry classification are shared. The source stepper
must outlive all predictions; closing a predictor never frees source state.
"""
import ctypes
import numpy as np

class PredictionStepper:
    def __init__(self,source,data):
        self.source=source;self.data=data;self.output=np.zeros(20,np.float64)
        if data.qpos.shape!=(source.model.nq,) or data.qvel.shape!=(source.model.nv,):
            raise ValueError('Prediction data dimensions must match the source model')
    def step(self,steps):
        s=self.source
        code=s.function(s.model._address,self.data._address,steps,s.payload,s.flex,s.head,
            s.classes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),s.joints.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),len(s.joints),
            self.output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if code:raise RuntimeError('Prediction step rejected its arguments')
        return self.output.copy()
    def close(self):pass
