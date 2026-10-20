"""Native acceleration must preserve motor-only actuation and failure evidence.

Intentional state/force corruption below belongs only to negative tests, never
to the runtime controller. No positive result is inferred from these tests.
"""
import unittest
import mujoco
import numpy as np
from native_household import NativeHousehold
from run_native_household import ControllerBridge


class NativePhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge=ControllerBridge()
        init=cls.bridge.request(op='init',config={'kind':'pen','approach':True})
        cls.runtime=NativeHousehold(init['xml'],init['config'])

    @classmethod
    def tearDownClass(cls):
        cls.runtime.close();cls.bridge.close()

    def setUp(self):
        self.r=self.runtime;self.r.reset({'rootOffset':[0,0,0]})

    def test_all_320_substeps_are_audited(self):
        row=self.r.step(self.r.data.ctrl.copy())
        self.assertEqual(self.r.summary['steps'],320)
        self.assertEqual(row['contact']['samples'],320)
        self.assertAlmostEqual(row['time'],.02,places=12)
        self.assertFalse(self.r.summary['solverReset'])

    def test_invalid_action_is_atomic(self):
        before=self.r.data.ctrl.copy();time=self.r.data.time
        for values in [before[:-1],np.r_[before[:-1],np.nan],np.r_[before[:-1],np.inf]]:
            with self.assertRaises(ValueError):self.r.step(values)
            np.testing.assert_array_equal(before,self.r.data.ctrl)
            self.assertEqual(time,self.r.data.time)

    def test_external_body_force_is_detected(self):
        self.r.data.xfrc_applied[self.r.payload_body,2]=.01
        self.r.step(self.r.data.ctrl.copy())
        self.assertEqual(self.r.summary['maxExternal'],.01)

    def test_external_joint_force_is_detected(self):
        self.r.data.qfrc_applied[self.r.va[5]]=.02
        self.r.step(self.r.data.ctrl.copy())
        self.assertEqual(self.r.summary['maxExternal'],.02)

    def test_collision_corruption_is_not_hidden(self):
        self.r.data.qpos[2]-=.10
        mujoco.mj_forward(self.r.model,self.r.data)
        self.r.step(self.r.data.ctrl.copy())
        self.assertGreater(self.r.summary['maxRobotPenetration'],.0005)

    def test_reset_clears_force_and_policy_history(self):
        self.r.data.xfrc_applied[self.r.payload_body,2]=.01
        self.r.last_action[:]=.3
        self.r.reset({'rootOffset':[.02,-.01,.1]})
        np.testing.assert_array_equal(self.r.last_action,np.zeros(14))
        np.testing.assert_array_equal(self.r.data.xfrc_applied,np.zeros_like(self.r.data.xfrc_applied))
        self.assertEqual(self.r.summary['maxExternal'],0)

    def test_head_jacobian_matches_finite_difference(self):
        r=self.r;kin=r.controller_context(True)['kinematics'];before=np.array(kin['headPoint'])
        eps=1e-7
        for j in [5,6,7,8]:
            q=r.data.qpos.copy();r.data.qpos[r.qa[j]]+=eps;mujoco.mj_forward(r.model,r.data)
            numerical=(r.data.xpos[r.head]-before)/eps
            np.testing.assert_allclose(numerical,np.array(kin['headPosition'])[:,j+6],atol=1e-7)
            r.data.qpos[:]=q;mujoco.mj_forward(r.model,r.data)

    def test_vertex_clearance_matches_collision_geometry(self):
        r=self.r;clearance=r.controller_context(True)['clearance']
        distance=mujoco.mj_geomDistance(r.model,r.data,r.model.geom('floor').id,clearance['geom'],1.,None)
        self.assertAlmostEqual(clearance['height'],distance,places=8)

    def test_model_has_no_attachments_or_pedestal(self):
        audit=self.r.model_audit
        self.assertTrue(audit['freeBase'] and audit['freePayload'])
        self.assertEqual(audit['equalities'],0);self.assertEqual(audit['mocapBodies'],0)
        self.assertFalse(audit['pedestalPresent'] or audit['terrainPresent'])
        self.assertEqual(audit['jawTorqueLimit'],[-.02,.02])


if __name__=='__main__':unittest.main()
