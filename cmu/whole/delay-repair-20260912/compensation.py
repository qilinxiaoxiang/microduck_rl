"""Causal compensation: only already-arrived 34D state, heading, and packet age.
No simulator, current pose, actuator state or truth-sensor references are accepted.
"""
import math
import numpy as np

def compensate(received,received_yaw,age_s,config):
    state=np.array(received,dtype=float,copy=True);yaw=float(received_yaw)
    if state.shape!=(34,):raise ValueError('Expected gyro3+gravity3+q14+dq14')
    mode=config.get('delay_compensation','none')
    if mode=='none' or age_s<=0:return state,yaw
    if mode!='constant_velocity':raise ValueError(mode)
    gain=float(config.get('prediction_fraction',1.0));h=float(age_s)*gain
    omega=state[:3].copy();g=state[3:6].copy()
    # Rodrigues exponential for g_dot = -omega cross g, expressed in body axes.
    angle=np.linalg.norm(omega)*h
    if angle>1e-12:
        axis=-omega/np.linalg.norm(omega)
        state[3:6]=g*math.cos(angle)+np.cross(axis,g)*math.sin(angle)+axis*np.dot(axis,g)*(1-math.cos(angle))
    state[6:20]+=state[20:34]*h
    if config.get('predict_heading',True):
        # ZYX yaw rate from delayed body angular velocity and delayed gravity.
        yaw_rate=(-g[1]*omega[1]-g[2]*omega[2])/max(g[1]**2+g[2]**2,.25)
        yaw+=float(np.clip(yaw_rate,-8,8))*h
        yaw=math.atan2(math.sin(yaw),math.cos(yaw))
    return state,yaw
