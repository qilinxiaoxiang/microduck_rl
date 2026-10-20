import copy
import unittest
from household_parameter_search import BOUNDS,CrossEntropySearch,decode,physical_score


class ParameterSearchTests(unittest.TestCase):
    def test_seed_reproducibility(self):
        self.assertEqual(CrossEntropySearch(7).ask(),CrossEntropySearch(7).ask())

    def test_bounds_and_physics_are_not_search_parameters(self):
        base={'mass':.006,'friction':.8,'jawTorque':.02,'dt':.0000625,'rootOffset':[.01,-.01,-.1]}
        for extreme in [-100,100]:
            candidate=decode([extreme]*len(BOUNDS),base)
            for key,value in base.items():self.assertEqual(candidate[key],value)
            self.assertEqual(candidate['gripOffset'][1],-.05)
            self.assertTrue(.04<=candidate['crouch']<=.22)
        with self.assertRaises(ValueError):decode([float('nan')]*len(BOUNDS),base)

    def test_elite_update_preserves_best(self):
        search=CrossEntropySearch(3)
        values=search.ask(4)
        search.tell([{'values':v,'score':i,'accepted':False} for i,v in enumerate(values)])
        self.assertEqual(search.ask(4)[0],values[3])
        self.assertTrue(all(s>=.05 for s in search.std))

    def test_safety_failure_dominates_apparent_success(self):
        summary=dict(finite=True,solverReset=False,maxExternal=0,nonFootSupport=False,maxPenetration=0,maxRobotPenetration=0,maxJointLimitExcess=0)
        record=dict(summary=summary,acceptance={'accepted':True},rows=[])
        for key,value in [('nonFootSupport',True),('maxExternal',.01),('maxPenetration',.0005),('maxRobotPenetration',.0005),('maxJointLimitExcess',.002)]:
            bad=copy.deepcopy(record);bad['summary'][key]=value
            self.assertLess(physical_score(bad),-900)

    def test_process_failure_cannot_be_rewarded(self):
        self.assertEqual(physical_score({'incomplete':True}),-10000)
        self.assertEqual(physical_score({'failure':'source mismatch'}),-10000)


if __name__=='__main__':unittest.main()
