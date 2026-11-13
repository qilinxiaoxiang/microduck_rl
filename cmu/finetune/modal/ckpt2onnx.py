"""ckpt2onnx.py <model_N.pt> <out.onnx> [template.onnx]: write the actor of an rsl_rl checkpoint into a copy of walk61.onnx
(same graph, metadata kept), or of the given template (a history checkpoint needs a history export of the same run,
e.g. its model_0.onnx; shapes are checked)."""
import sys, torch, onnx, numpy as np
from pathlib import Path
from onnx import numpy_helper
TEMPLATE=str(Path(__file__).resolve().parents[3] / 'cmu/whole/runtime-overlay/models/walk61.onnx')
if len(sys.argv) > 3: TEMPLATE=sys.argv[3]
ck=torch.load(sys.argv[1], map_location='cpu', weights_only=False)['actor_state_dict']
m=onnx.load(TEMPLATE)
EPS=1e-2
repl={}
for init in m.graph.initializer:
    n=init.name
    if n in ck:
        repl[n]=ck[n].numpy().astype(numpy_helper.to_array(init).dtype)
    elif n.startswith('onnx::Div'):
        repl[n]=(ck['obs_normalizer._std']+EPS).numpy().astype(np.float32)
    else:
        print('untouched initializer', n, numpy_helper.to_array(init).shape)
for init in m.graph.initializer:
    if init.name in repl:
        assert repl[init.name].shape == numpy_helper.to_array(init).shape, (init.name, repl[init.name].shape, numpy_helper.to_array(init).shape)
        init.CopyFrom(numpy_helper.from_array(repl[init.name], init.name))
onnx.save(m, sys.argv[2])
print('wrote', sys.argv[2], 'replaced', sorted(repl))
