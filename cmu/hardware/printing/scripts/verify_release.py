"""Read-only source -> native -> SD ZIP chain verification, including tamper tests."""
import argparse,json,math,zipfile
from pathlib import Path
import release_pipeline as r

def verify(run):
 d=Path(run).resolve();pkg=r.read(d/'package-receipt.json');indexp=r.bound(pkg['index']);idx=r.read(indexp);zpath=r.bound(pkg['zip']);prep_path=r.bound(idx['preparation']);prep=r.read(prep_path);r.bound(prep['pipeline']);manifest=r.bound(idx['manifest']);assert prep['manifest']==idx['manifest'];m,checked=r.source_receipt(manifest);assert checked['parts']==prep['parts']
 audit=r.read(r.bound(idx['incrementalAudit']));assert audit['manifest']==idx['manifest'] and audit['parts']==prep['parts'] and audit['status']==idx['status']
 checks=r.read(r.bound(idx['nativeChecks']));assert checks['passed'] and checks['preparation']==idx['preparation']
 assert len(idx['files'])==len(checks['checks'])==len(prep['rows'])==2*len({p['color']for p in m['parts']})
 if idx['status']=='RELEASED':
  assert m['releaseKind']=='incremental';review=r.read(r.bound(idx['nativeReview']));assert review['passed'] and review['preparationSha256']==r.sha(prep_path)
 if m['releaseKind']=='fixture':assert idx['status']=='FIXTURE_ONLY_NOT_RELEASED'
 with zipfile.ZipFile(zpath)as z:
  assert sorted(z.namelist())==sorted(f['filename']for f in idx['files'])
  for f,c,row in zip(idx['files'],checks['checks'],prep['rows']):
   source=r.bound(c['nativeArchive']);res=r.read(r.bound(c['result']));assert res['archive']==c['nativeArchive'] and res['preparation']==idx['preparation'];r.bound(res['project']);r.bound(row['profile'])
   for b in row['supportBlockers']:r.bound(b)
   command=r.read(r.bound(res['command']));assert command['project']==row['project']and command['preparation']==idx['preparation']
   for ext in command['extraProfiles']:r.bound(ext)
   g,cfg,imgs=r.native_payload(source);assert r.digest(g)==c['nativeCommandStreamSha256']==res['gcodeSha256'];r.check_profile(cfg,c['machine'])
   assert r.time_seconds(g,c['machine'])==f['estimatedSeconds'];minutes=math.ceil(f['estimatedSeconds']/60);label=(f'{minutes//60}h'if minutes>=60 else'')+f'{minutes%60}m';assert f['estimatedTime']==label
   assert f['filename']==f'zhengxia_{m["version"]}_{f["color"]}_{r.MACHINES[c["machine"]]["slug"]}_{label}'+('.gcode.3mf'if c['machine']=='Bambu'else'.gcode')
   assert r.sha(f['path'])==f['sha256']==r.digest(z.read(f['filename']))
   assert c['nativeFootprint']['insideBed'] and all(x['model_area_mm2']>0 for x in c['nativeFirstLayer'])
   preview=r.bound(c['thumbnailSource']);_,_,images=r.native_payload(preview);assert r.digest(images['Metadata/plate_1.png'])==c['thumbnailSha256']
   if c['machine']=='Bambu':assert r.sha(f['path'])==r.sha(source)
   else:r.check_cc_preview(Path(f['path']).read_bytes(),g)
   if idx['status']=='RELEASED':assert review['nativeArchives'][str(source)]==r.sha(source)
 return dict(passed=True,status=idx['status'],files=len(idx['files']),manifest=r.ref(manifest),index=r.ref(indexp),zip=r.ref(zpath),pipeline=r.ref(r.__file__),verifier=r.ref(__file__),noPhysicalPrinterExecution=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--output');a=ap.parse_args();out=verify(a.run)
 if a.output:r.write(a.output,out)
 print(json.dumps(out,indent=2))
