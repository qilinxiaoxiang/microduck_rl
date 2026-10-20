"""Compile a small, audited C++ stepping loop against the installed MuJoCo.

Build products are private temporary files; the simulator source and Python
environment are not modified. A single process owns each build directory.
"""
import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile
import mujoco
import numpy as np


class ContactStepper:
    def __init__(self, model, data, joint_names):
        base = Path(mujoco.__file__).parent
        library = next(base.glob("libmujoco*"))
        self.build = tempfile.TemporaryDirectory(prefix="microduck-native-step-")
        output = Path(self.build.name) / ("step.dylib" if sys.platform == "darwin" else "step.so")
        source = Path(__file__).with_suffix(".cpp")
        subprocess.run(["c++", "-std=c++17", "-O3", "-dynamiclib" if sys.platform == "darwin" else "-shared",
                        "-fPIC", "-I" + str(base / "include"), str(source), str(library),
                        "-Wl,-rpath," + str(base), "-o", str(output)], check=True, capture_output=True, text=True)
        self.library = ctypes.CDLL(str(output))
        self.function = self.library.microduck_step
        ints = ctypes.POINTER(ctypes.c_int)
        self.function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ints, ints,
                                  ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self.function.restype = ctypes.c_int
        self.model, self.data = model, data
        name_id = lambda kind, name: mujoco.mj_name2id(model, kind, name)
        self.payload = name_id(mujoco.mjtObj.mjOBJ_GEOM, "payload_geom")
        self.flex = name_id(mujoco.mjtObj.mjOBJ_FLEX, "payload_flex")
        self.head = model.body("jaw_soft").id
        trunk = model.body("trunk_base").id
        self.classes = np.zeros(model.ngeom, np.int32)
        for geom in range(model.ngeom):
            body = model.geom_bodyid[geom]
            while body > 0 and body != trunk:
                body = model.body_parentid[body]
            if body == trunk:
                self.classes[geom] |= 1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if name in ["left_foot_collision", "right_foot_collision"]:
                self.classes[geom] |= 2
            if name.startswith(("beak_jaw_contact_", "beak_jaw_soft_contact_")):
                self.classes[geom] |= 4
            if name.startswith("beak_soft_mouth_top_contact_"):
                self.classes[geom] |= 8
            if name.startswith(("beak_bottom_head_shell_contact_", "beak_top_head_shell_contact_")):
                self.classes[geom] |= 16
        self.joints = np.array([model.joint(name).id for name in [*joint_names, "beak_joint"]], np.int32)
        self.output = np.zeros(20, np.float64)

    def step(self, steps):
        code = self.function(self.model._address, self.data._address, steps, self.payload,
                             self.flex, self.head, self.classes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                             self.joints.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(self.joints),
                             self.output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if code:
            raise RuntimeError("Native step rejected its arguments")
        return self.output.copy()

    def close(self):
        self.build.cleanup()
