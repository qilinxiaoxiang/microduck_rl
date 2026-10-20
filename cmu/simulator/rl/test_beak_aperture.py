import unittest
import numpy as np
from inspect_beak_aperture import ray_interval, inspect
from run_native_household import ControllerBridge


class BeakApertureTests(unittest.TestCase):
    def test_cube_ray_and_miss(self):
        vertices=np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                           [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], dtype=float)
        faces=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                        [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]])
        self.assertEqual(ray_interval(vertices,faces,0,0),[-1,1])
        self.assertIsNone(ray_interval(vertices,faces,2,0))
        self.assertEqual(ray_interval(vertices,faces[:,::-1],0,0),[-1,1])

    def test_actual_pad_coverage_and_clearance(self):
        bridge=ControllerBridge()
        try:
            initial=bridge.request(op='init',config={'kind':'sock'})
            result=inspect(initial['xml'],initial['config'])
        finally:
            bridge.close()
        for ray in result['rays']:
            if ray['y']==-.05:
                self.assertIsNone(ray['gap'])
            else:
                self.assertGreater(ray['gap'],.0005)
                self.assertLess(ray['gap'],.0008)


if __name__=='__main__':unittest.main()
