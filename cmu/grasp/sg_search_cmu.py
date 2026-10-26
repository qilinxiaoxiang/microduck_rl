"""Run Pollen's static grasp pose search (scripts/static_grasp/sg_search.py) on a CMU task's model.

  PYTHONPATH=src python cmu/grasp/sg_search_cmu.py --task Mjlab-StaticGrasp-Hover-FromPick-MicroDuck-CMU-Plugged-Ball \
      -- --family sym --mode lips --h 1 --name cmu_sym_lips_1
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "static_grasp"))
argv = sys.argv[1:]
task = argv[argv.index("--task") + 1]
rest = argv[argv.index("--") + 1:]
import sg_model
sg_model.TASK = task
sg_model.build_model.__defaults__ = (task,)
sg_model.StaticModel.__init__.__defaults__ = (task,)
import sg_search
sg_search.OUT = HERE / "poses"
sys.argv = ["sg_search.py"] + rest
assert sg_model.StaticModel().m.nbody > 0 and sg_model.TASK == task
print('[sg_search_cmu] model task', task, 'mass g', round(float(sg_model.StaticModel().m.body_mass.sum()) * 1000, 1), flush=True)
sg_search.main()
