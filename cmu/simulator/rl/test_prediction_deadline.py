"""Deadline failures must leave the actual robot unchanged and close workers."""
import time
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from grasp_forecast import check_prediction_deadline,forecast_grasp
from predictive_reach import PredictiveReach
from native_household import NativeHousehold
from run_native_household import ControllerBridge


class DeadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge=ControllerBridge();initial=cls.bridge.request(op='init',config={'kind':'pen'})
        cls.r=NativeHousehold(initial['xml'],initial['config'])

    @classmethod
    def tearDownClass(cls):
        cls.r.close();cls.bridge.close()

    def state(self):
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.zeros(mujoco.mj_stateSize(self.r.model,spec))
        mujoco.mj_getState(self.r.model,self.r.data,state,spec)
        return state

    def test_invalid_deadlines_fail_closed(self):
        for value in [float('nan'),float('inf'),True,'tomorrow']:
            with self.assertRaises(ValueError):check_prediction_deadline(value)

    def test_expired_prediction_does_not_advance_actual_physics(self):
        before=self.state();p=PredictiveReach(self.r,deadline=time.monotonic()-1)
        try:
            with self.assertRaises(TimeoutError):p.update()
        finally:p.close()
        np.testing.assert_array_equal(before,self.state())
        self.assertEqual(p.updates,0)

    def test_closure_checks_deadline_after_each_real_control_interval(self):
        before=self.state()
        # The independent copy really takes one 320-substep control interval.
        with patch('grasp_forecast.time.monotonic',side_effect=[1.,1.,3.]):
            with self.assertRaises(TimeoutError):forecast_grasp(self.r,self.r.data.ctrl[:14].copy(),deadline=2.)
        np.testing.assert_array_equal(before,self.state())

    def test_running_parallel_forecasts_drain_at_deadline(self):
        before=self.state();start=time.monotonic();p=PredictiveReach(self.r,deadline=start+.02)
        try:
            with self.assertRaises(TimeoutError):p.update()
        finally:p.close()
        np.testing.assert_array_equal(before,self.state())
        self.assertLess(time.monotonic()-start,10)


if __name__=='__main__':unittest.main()
