"""Selection unit tests; synthetic forecasts are not physical acceptance."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from predictive_reach import PredictiveReach

class CandidateTests(unittest.TestCase):
    def make_planner(self,allow_escape):
        p=PredictiveReach.__new__(PredictiveReach)
        p.position_weight=1.;p.velocity_seconds=0.;p.stage='reach'
        p.r=SimpleNamespace(model=None,data=SimpleNamespace(time=1.,ctrl=np.zeros(15)),config={})
        p.ctrl=np.zeros(14);p.lower=np.full(14,-1.);p.upper=np.full(14,1.)
        p.last_plan=None;p.last_time=-1;p.updates=0;p.total_seconds=0
        p.pool=ThreadPoolExecutor(max_workers=2);self.addCleanup(p.close)
        p.goal=lambda point:np.r_[.1,np.zeros(19)]
        def predict(ctrl):
            safe=allow_escape and abs(ctrl[0]-.025)<1e-9 and np.count_nonzero(ctrl)==1
            return dict(features=np.r_[ctrl[0],np.zeros(19)],height=.005,unsafe=not safe,physical={})
        p.predict=predict
        return p

    def evaluate(self,p):
        with patch('mujoco.mj_stateSize',return_value=2),patch('mujoco.mj_getState'):
            return p.update()

    def test_reuses_safe_finite_difference_candidate(self):
        result=self.evaluate(self.make_planner(True))
        self.assertFalse(result['blocked']);self.assertEqual(result['audit']['safeCandidates'],1)
        self.assertEqual(result['audit']['candidate'],5)
        self.assertAlmostEqual(result['ctrl'][0],.025)

    def test_no_safe_candidate_preserves_targets_and_blocks(self):
        result=self.evaluate(self.make_planner(False))
        self.assertTrue(result['blocked']);self.assertEqual(result['ctrl'],[0.]*14)
        self.assertEqual(result['audit']['safeCandidates'],0)

    def test_cost_parameter_bounds(self):
        for config in [dict(predictionHorizon=.1),dict(predictionHorizon=.7),dict(predictionPositionWeight=0),dict(predictionPositionWeight=5),dict(predictionVelocitySeconds=-.1),dict(predictionVelocitySeconds=.4),dict(predictionPositionWeight=float('nan'))]:
            with self.assertRaises(ValueError):PredictiveReach(SimpleNamespace(config=config))

    def test_transport_requires_preceding_reach(self):
        p=self.make_planner(True);p.ctrl=None
        with self.assertRaisesRegex(ValueError,'preceding reach'):p.update('transport')
        self.assertEqual(p.stage,'reach')

    def test_material_node_is_bounded_and_cannot_switch(self):
        p=self.make_planner(True);p.cloth_node=None
        p.r.config={'kind':'sock','raisedCloth':True};p.r.flex=0
        p.r.model=SimpleNamespace(flex_vertnum=[2])
        for invalid in [-1,2,True,1.5]:
            with self.assertRaises(ValueError):p.set_cloth_node(invalid)
        p.set_cloth_node(1);p.set_cloth_node(1)
        with self.assertRaisesRegex(ValueError,'retain'):p.set_cloth_node(0)
        self.assertEqual(p.cloth_node,1)

    def test_material_goal_uses_selected_vertex_not_toe(self):
        p=self.make_planner(True);del p.goal
        p.cloth_node=None;p.reference=np.zeros(20)
        s=dict(payload=[0,0,.002,1,0,0,0],clothPoints=[[0,0,.002],[.01,.02,.006]])
        p.r.config={'kind':'sock','raisedCloth':True};p.r.snapshot=lambda:s
        with self.assertRaisesRegex(ValueError,'Missing tracked'):p.goal(np.zeros(3))
        p.cloth_node=1
        np.testing.assert_array_equal(p.goal(np.zeros(3))[:3],s['clothPoints'][1])
        self.assertEqual(s['payload'][2],.002)

    def test_grip_features_follow_controller_local_offset(self):
        p=self.make_planner(True);p.position_weight=2;p.grip_offset=np.array([-.001,-.05,.003])
        p.r.head=0;p.r.trunk=0;p.r.feet=[0,1]
        data=SimpleNamespace(xmat=np.eye(3).reshape(1,9),xpos=np.zeros((1,3)),geom_xpos=np.zeros((2,3)),subtree_com=np.zeros((1,3)))
        np.testing.assert_allclose(p.features(data)[:3],2*np.array([-.017,-.05,-.072]))
        p.grip_offset=np.array([.005,-.035,.0005])
        np.testing.assert_allclose(p.features(data)[:3],2*np.array([-.011,-.035,-.0745]))
        np.testing.assert_array_equal(data.xpos,np.zeros((1,3)))

    def test_transport_phase_uses_measured_goal_and_bypasses_cache(self):
        p=self.make_planner(True);p.reference=np.arange(20,dtype=float)
        p.features=lambda data:np.arange(20,dtype=float)+10
        p.last_plan={'stage':'reach'};p.last_time=p.r.data.time
        with patch('mujoco.mj_stateSize',return_value=2),patch('mujoco.mj_getState'):
            result=p.update('transport')
        self.assertEqual(result['stage'],'transport');self.assertEqual(p.jaw,0)
        self.assertAlmostEqual(p.transport_goal[2],12.07)
        np.testing.assert_array_equal(p.transport_goal[12:20],p.reference[12:20])
        self.assertEqual(p.updates,1)
        with self.assertRaisesRegex(ValueError,'new planner'):p.update('reach')

if __name__=='__main__':unittest.main()
