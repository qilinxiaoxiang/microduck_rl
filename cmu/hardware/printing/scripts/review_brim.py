"""Per-object brim review of one native slice against a per-part brim plan (pipeline v2). Read-only on its inputs.

    /tmp/microduck-cad-venv/bin/python review_brim.py RUN_DIR COLOR MACHINE [--plan PREPARATION.json] [--output OUT.json]

RUN_DIR is a pipeline run (preparation.json, <Color>_<Machine>_editable.3mf, native/<Color>_<Machine>/sliced.3mf). The plan
defaults to RUN_DIR/preparation.json and must come from release_pipeline_v2 (parts carry objectSettings). --plan lets an
older slice, such as a v1 release, be checked against a v2 plan for the same parts and copies. Exit status 0 = pass, 1 = fail.
A release review script calls release_pipeline_v2.brim_review(first_layer_rows, parts) the same way.
"""
import argparse, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
import release_pipeline_v2 as r


def review(run, color, machine, plan=None):
    run = Path(run).resolve(); prep = r.read(run / 'preparation.json'); planp = Path(plan or run / 'preparation.json').resolve(); planned = r.read(planp)
    row = next(x for x in prep['rows'] if x['color'] == color and x['machine'] == machine); project = r.bound(row['project'])
    res = r.read(run / f'native/{color}_{machine}/result.json'); arch = r.bound(res['archive'])
    g, cfg, _ = r.native_payload(arch); assert r.digest(g) == res['gcodeSha256']; r.check_profile(cfg, machine)
    rows, _ = r.first_layer(g, project, machine)
    out = r.brim_review(rows, [p for p in planned['parts'] if p['color'] == color], machine)
    out.update(color=color, archive=r.ref(arch), project=r.ref(project), plan=r.ref(planp), pipeline=r.ref(r.__file__),
               plateBrimSettings={k: cfg.get(k) for k in ('brim_type', 'brim_width', 'brim_object_gap', 'skirt_loops', 'raft_layers')})
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('color'); ap.add_argument('machine', choices=r.MACHINES)
    ap.add_argument('--plan'); ap.add_argument('--output'); a = ap.parse_args()
    o = review(a.run, a.color, a.machine, a.plan)
    if a.output: r.write(a.output, o)
    print(json.dumps(dict(passed=o['passed'], color=a.color, machine=a.machine, plate=o['plateBrimSettings'])))
    for k, v in o['objects'].items():
        print(f"   {k}: {v['brimType']} brim {v['brimAreaMm2']} mm2 (model {v['modelAreaMm2']}), inside own hull {v['brimInsideOwnHullMm2']}, reach {v['brimReachMm']}",
              '-> PASS' if v['passed'] else '-> FAIL: ' + '; '.join(v['failures']))
    sys.exit(0 if o['passed'] else 1)
