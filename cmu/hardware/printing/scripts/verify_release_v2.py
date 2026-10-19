"""Read-only verification of a pipeline-v2 run: the v1 source -> native -> SD ZIP chain (verify_release.py, run against
release_pipeline_v2) plus the per-object brim review recomputed from each native archive.

    /tmp/microduck-cad-venv/bin/python verify_release_v2.py RUN_DIR [--output OUT.json]

Runs prepared by release_pipeline.py (v1: V1.4-V1.11) are verified with verify_release.py, which stays unchanged.
"""
import argparse,json
from pathlib import Path
import release_pipeline_v2 as r
import verify_release as base

def verify(run):
 d=Path(run).resolve();idx=r.read(r.bound(r.read(d/'package-receipt.json')['index']));prep=r.read(r.bound(idx['preparation']))
 assert prep.get('pipelineVersion')==2 and Path(prep['pipeline']['path'])==Path(r.__file__).resolve(),'Not a pipeline v2 run; use verify_release.py'
 v1=base.r;base.r=r   # same chain checks, bound to the v2 module and its source_receipt
 try:out=base.verify(d)
 finally:base.r=v1
 checks=r.read(r.bound(idx['nativeChecks']));brims={}
 for c,row in zip(checks['checks'],prep['rows']):
  assert (c['color'],c['machine'])==(row['color'],row['machine'])
  g,_,_=r.native_payload(r.bound(c['nativeArchive']));rows,_=r.first_layer(g,r.bound(row['project']),c['machine'])
  review=r.brim_review(rows,[p for p in prep['parts']if p['color']==c['color']],c['machine'])
  assert review==c['brimReview'],f'{c["color"]} {c["machine"]}: recomputed brim review differs from the packaged one'
  assert review['passed'],f'{c["color"]} {c["machine"]}: brim review failed'
  brims[f'{c["color"]}_{c["machine"]}']={k:dict(brimType=v['brimType'],brimAreaMm2=v['brimAreaMm2'],passed=v['passed'])for k,v in review['objects'].items()}
 out.update(pipeline=r.ref(r.__file__),verifier=r.ref(__file__),baseVerifier=r.ref(base.__file__),brimReview=brims)
 return out

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--output');a=ap.parse_args();out=verify(a.run)
 if a.output:r.write(a.output,out)
 print(json.dumps(out,indent=2))
