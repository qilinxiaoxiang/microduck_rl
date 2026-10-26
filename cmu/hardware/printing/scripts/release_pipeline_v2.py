"""Manifest-driven incremental native print preparation, pipeline v2. No printer submission API.

v2 adds per-object slicer settings (brim per part). release_pipeline.py is the v1 pipeline: every V1.4-V1.11 receipt binds
its exact bytes, so it stays frozen. Differences from v1:
- part rows carry `objectSettings` (brim_type, brim_width, brim_object_gap). Without it the R1 default applies: tip angle
  below 10 degrees gets an outer 4 mm brim, every other part no_brim. Brim keys in plate-wide processOverrides are rejected.
- prepare() writes each object's settings into Metadata/model_settings.config; preparation.json records them per part.
- first_layer() measures brim per object (Orca prints brim outside the object labels; it is assigned to the nearest object).
- a row may set `colourOnlyReprint` (a reason) to reprint the exact stock STL of the instances it replaces in another colour;
  `damageReprint` (a reason) does the same in the same colour for a printed part that broke or bent. preparation.json counts
  both in reprintUnchangedStockQuantity. Every other row still needs changed geometry.
- brim_review() fails a part with an unplanned brim, a planned brim that is missing or of another width than planned, or brim
  inside its own convex outline (an open opening filled with brim); package() requires it to pass on every plate.
- source_receipt() runs wall_check.check() on every part in its print frame: a wall under 1.2 mm over 2 mm2 or more stops
  prepare unless the row's thinWallAllowances covers it; preparation.json records the result per part under wallCheck.
"""
from pathlib import Path
import argparse,base64,copy,hashlib,io,json,math,re,shutil,struct,subprocess,sys,zipfile
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import wall_check

HERE=Path(__file__).resolve().parent
PRINT=HERE.parent
REDESIGN=PRINT.parent
OLD=PRINT/'12-REPAIR-V1-20261008/dual-printer'
NS='http://schemas.microsoft.com/3dmanufacturing/core/2015/02'
Q='{'+NS+'}'
ET.register_namespace('',NS)
COLORS=('Yellow','White','Orange','Black')
MACHINES={
 'Bambu':dict(label='Bambu X1 Carbon',slug='Bambu-X1C',model='Bambu Lab X1 Carbon',bed='Cool Plate',material='GFA00',exe='/Applications/BambuStudio.app/Contents/MacOS/BambuStudio'),
 'Elegoo':dict(label='Elegoo Centauri Carbon',slug='Elegoo-Centauri-Carbon',model='Elegoo Centauri Carbon',bed='Textured PEI Plate',material='EPLAB00',exe='/Volumes/OrcaSlicer/OrcaSlicer.app/Contents/MacOS/OrcaSlicer')}

# Per-object settings. Only keys that both Bambu Studio and OrcaSlicer read per object from model_settings.config and that
# the native review checks are accepted.
BRIM_KEYS=('brim_type','brim_width','brim_object_gap')
OBJECT_KEYS=set(BRIM_KEYS)
# The review defines pass/fail for these two only: inner and ear brims, and auto_brim (the slicer decides), are not accepted.
BRIM_TYPES={'no_brim','outer_only'}
# guide/print/orientation-study/PRINT-ORIENTATION-PRACTICE.md R1: a 3 to 5 mm brim below a 10 degree tip angle.
R1_TIP_DEG=10.0
R1_BRIM={'brim_type':'outer_only','brim_width':'4','brim_object_gap':'0.1'}
R1_NO_BRIM={'brim_type':'no_brim'}
# Tip angle: same definition and constants as guide/print/orientation-study/orientation_check.py metrics().
TIP_CONTACT_TOL=.01;TIP_SECTION_Z=.1
# Brim review tolerances (first layer, extrusion paths buffered to their own line width).
BRIM_INSIDE_HULL_MAX_MM2=2.0   # brim area allowed inside the part's own convex first-layer outline
BRIM_REACH_TOL_MM=.5           # measured outer reach vs brim_object_gap + brim_width (V1.9 head cap: 4.29 for 4.1 planned;
                               # the Orange donor default brim_width 5 would read about 5.3)

def tip_angle(mesh):
 """Tip angle in degrees of a print-frame mesh resting on z=0 (orientation_check.py metrics(): convex hull of true contact
 plus the z=0.1 mm section, centre of mass, atan2(margin, com height)). <= 0 means not self-standing."""
 from shapely.geometry import MultiPoint,Point
 tri=mesh.triangles;contact=tri[:,:,2].max(1)<=TIP_CONTACT_TOL;pts=tri[contact].reshape(-1,3)[:,:2]
 seg=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,TIP_SECTION_Z])
 if len(seg):pts=np.r_[pts.reshape(-1,2),seg.reshape(-1,3)[:,:2]]
 if len(pts)<3:pts=mesh.vertices[mesh.vertices[:,2]<=.1][:,:2]
 com=mesh.center_mass if mesh.is_volume else mesh.centroid;margin=-math.inf
 if len(pts)>=3:
  hull=MultiPoint([tuple(p)for p in pts]).convex_hull
  if hull.geom_type=='Polygon':
   d=hull.exterior.distance(Point(com[0],com[1]));margin=d if hull.contains(Point(com[0],com[1]))else -d
 return math.degrees(math.atan2(margin,max(com[2],1e-6)))if math.isfinite(margin)else -90.

def object_settings(row,tip):
 """Effective per-object settings and their source: explicit manifest objectSettings (with objectSettingsReason), else R1."""
 if 'objectSettings' in row:
  s=row['objectSettings'];assert isinstance(s,dict) and s,f'{row["key"]}: objectSettings must be a non-empty object'
  assert set(s)<=OBJECT_KEYS,f'{row["key"]}: unsupported per-object keys {sorted(set(s)-OBJECT_KEYS)}'
  assert all(isinstance(v,str)for v in s.values()),f'{row["key"]}: per-object values are slicer strings'
  assert s.get('brim_type')in BRIM_TYPES,f'{row["key"]}: objectSettings needs a brim_type from {sorted(BRIM_TYPES)}'
  if s['brim_type']!='no_brim':
   assert {'brim_width','brim_object_gap'}<=set(s),f'{row["key"]}: a planned brim needs brim_width and brim_object_gap'
   assert float(s['brim_width'])>0 and float(s['brim_object_gap'])>=0
  assert str(row.get('objectSettingsReason','')).strip(),f'{row["key"]}: explicit objectSettings need objectSettingsReason'
  return dict(s),'manifest: '+row['objectSettingsReason'].strip()
 if tip<R1_TIP_DEG:return dict(R1_BRIM),f'R1 default: tip angle {tip:.1f} deg < {R1_TIP_DEG:g} deg, outer brim'
 return dict(R1_NO_BRIM),f'R1 default: tip angle {tip:.1f} deg >= {R1_TIP_DEG:g} deg, no brim'

def planned_brim(settings):return settings.get('brim_type','no_brim')!='no_brim'

def hull_allowance(row):
 """A manifest row may raise the brim-inside-own-outline limit for a real recess in its bed face (V1.22 ankle bases: the
 stock 12 x 1.1 mm notch in the gusset end face takes 13.3 mm2 of Bambu's outer brim). Needs a written reason; never for
 an opening filled with brim."""
 if 'brimInsideHullAllowanceMm2' not in row:return {}
 a=float(row['brimInsideHullAllowanceMm2']);assert a>BRIM_INSIDE_HULL_MAX_MM2,f'{row["key"]}: brimInsideHullAllowanceMm2 must exceed {BRIM_INSIDE_HULL_MAX_MM2}'
 assert str(row.get('brimInsideHullAllowanceReason','')).strip(),f'{row["key"]}: brimInsideHullAllowanceMm2 needs brimInsideHullAllowanceReason'
 return dict(brimInsideHullAllowanceMm2=a,brimInsideHullAllowanceReason=row['brimInsideHullAllowanceReason'].strip())

def support_brim(row):
 """A manifest row whose first layer carries support next to the part (V1.24 ankle inner halves: the support under the
 foot-block arm stands on the bed beside the heel chamfer) may state brimAlongSupportReason. Both slicers then also brim
 that support (Orca with its own tree-support brim width), so the reach from the part alone runs to the far side of the
 support. brim_review then judges only the part's own brim, the brim outline points nearer the part than its first-layer
 support: their reach must not exceed gap + width + tolerance. It cannot require the full width there, because Bambu
 trims the part's brim where support stands closer than gap + width (V1.24 inner halves: 2.7 mm of a planned 4.1 mm)."""
 if 'brimAlongSupportReason' not in row:return {}
 why=str(row['brimAlongSupportReason']).strip();assert why,f'{row["key"]}: brimAlongSupportReason is empty'
 return dict(brimAlongSupportReason=why)

def wall_result(row,mesh,legacy):
 """Wall check of a print-frame part. Fails the row unless the manifest carries wallCheckLegacy (fixture manifests only:
 geometry printed before the wall rule, kept as a test of other checks); the result is recorded either way."""
 w=wall_check.check(mesh,wall_check.allowances(row),row['key']);failures=w.pop('failures')
 if legacy:w['legacy']=legacy
 else:assert not failures,'; '.join(failures)
 return w

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2)+'\n')
def ref(p):p=Path(p).resolve();return {'path':str(p),'sha256':sha(p)}
def bound(r):
 p=Path(r['path']).resolve();assert p.is_file(),str(p);assert sha(p)==r['sha256'],f'Source changed: {p}';return p

def load_mesh(p,T=None):
 m=trimesh.load(p,force='mesh',process=False)
 # Exact vertex deduplication is required: per-triangle STL vertices produced a
 # false, almost empty first layer in the earlier native neck export.
 v,inv=np.unique(m.vertices,axis=0,return_inverse=True)
 m=trimesh.Trimesh(v,inv[m.faces],process=False)
 assert m.is_watertight and m.is_winding_consistent,f'Nonclosed source {p}'
 if T is not None:m.apply_transform(T)
 return m

def pose_matrix(row):
 T=np.array(row['sourceToPrint'],dtype=float)
 assert T.shape==(4,4) and np.allclose(T[3],[0,0,0,1],atol=1e-10)
 assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8) and abs(np.linalg.det(T[:3,:3])-1)<1e-8,'Only rigid rotations/translations permitted'
 return T

def source_receipt(manifest_path):
 p=Path(manifest_path).resolve();m=read(p)
 assert m['schemaVersion']==1 and m['releaseKind'] in ('incremental','fixture')
 assert re.fullmatch(r'[A-Za-z0-9._-]+',m['version'])
 baseline=bound(m['baselineStock'])
 records=[r for r in read(baseline)['copy_records'] if 'manifest_instance_index' in r]
 assert len(records)==89 and len({r['manifest_instance_index'] for r in records})==89
 stock={str(r['object_id']):r for r in records}
 stock_hashes=set()
 for r in records:
  assert sha(r['source'])==r['sha256'] and sha(PRINT/r['copy'])==r['sha256'],'89-stock source drift'
  stock_hashes.add(r['sha256'])
 keys=set();replaced=set();parts=[];unchanged=0
 legacy=str(m.get('wallCheckLegacy','')).strip()
 assert not legacy or m['releaseKind']=='fixture','wallCheckLegacy is honoured only in fixture manifests'
 for row in m['parts']:
  assert row['key'] not in keys;keys.add(row['key'])
  assert re.fullmatch('[A-Za-z0-9_-]+',row['key']) and row['color'] in COLORS
  assert type(row['quantity']) is int and row['quantity']>0
  source=bound(row);T=pose_matrix(row);mesh=load_mesh(source,T)
  ids=list(map(str,row['replacesStockIds']))
  assert len(ids)==len(set(ids)) and all(x in stock for x in ids),'Unknown/duplicate stock object ID'
  assert not replaced.intersection(ids),'Two replacement rows target the same stock instance';replaced.update(ids)
  assert row['decision'] in ('addition','replacement')
  assert bool(ids)==(row['decision']=='replacement')
  assert len(ids)<=row['quantity'],'More replaced instances than replacement quantity'
  # A colour-only reprint names the exact stock STL of every instance it replaces and says why the colour changes;
  # every other row must bring changed geometry.
  colour_only=str(row.get('colourOnlyReprint','')).strip();damage=str(row.get('damageReprint','')).strip()
  assert not(colour_only and damage),f'{row["key"]}: colourOnlyReprint and damageReprint are exclusive'
  if colour_only or damage:
   assert row['decision']=='replacement' and all(stock[x]['sha256']==row['sha256']for x in ids),f'{row["key"]}: a colour-only or damage reprint must be the exact stock STL of each replaced instance'
   unchanged+=row['quantity']
  else:assert row['sha256'] not in stock_hashes,'Unchanged 89-stock STL requested for reprint'
  # A source-bound prior audit or explicit change reason describes a genuine
  # geometry change. Hash inequality alone is not geometry equivalence proof.
  assert row['changeReason'].strip()
  evidence=bound(row['incrementalEvidence'])
  minimum=float(mesh.bounds[0,2]);delta=np.array([0.,0.,-minimum]);mesh.apply_translation(delta)
  verts=mesh.triangles
  at_bottom=np.max(np.abs(verts[:,:,2]),axis=1)<=.01
  bottom=float(mesh.area_faces[at_bottom].sum())
  assert bottom>0,f'{row["key"]}: no real planar contact within 0.01 mm'
  assert bottom>=float(row['flatContact']['minimumAreaMm2']),f'{row["key"]}: intended bottom missing'
  b=mesh.bounds.tolist();tip=tip_angle(mesh);settings,settings_source=object_settings(row,tip)
  parts.append(dict(key=row['key'],source=ref(source),quantity=row['quantity'],color=row['color'],decision=row['decision'],replacesStockIds=ids,changeReason=row['changeReason'],incrementalEvidence=ref(evidence),sourceToPrint=T.tolist(),dropToBedMm=float(delta[2]),flatContactAreaMm2=bottom,bounds=b,vertexCount=len(mesh.vertices),triangleCount=len(mesh.faces),tipAngleDeg=round(tip,2),objectSettings=settings,objectSettingsSource=settings_source,**({'colourOnlyReprint':colour_only}if colour_only else {}),**({'damageReprint':damage}if damage else {}),**hull_allowance(row),**support_brim(row),wallCheck=wall_result(row,mesh,legacy)))
 assert parts
 # The brim is a per-part decision in v2; a plate-wide brim would reach every object that lacks its own value.
 for k,v in m.get('processOverrides',{}).items():
  assert not set(v)&set(BRIM_KEYS),f'processOverrides[{k}] sets {sorted(set(v)&set(BRIM_KEYS))}: set the brim per part (objectSettings)'
 return m,dict(schemaVersion=1,pipelineVersion=2,manifest=ref(p),pipeline=ref(__file__),wallCheckModule=ref(wall_check.__file__),baselineStock=ref(baseline),baselinePrintedStockInstances=89,reprintUnchangedStockQuantity=unchanged,partTypes=len(parts),printQuantity=sum(r['quantity']for r in parts),parts=parts,claimBoundary='Exact source identity and source-bound incremental decisions; no physical stock recount or manufacturing qualification.')

def check_profile(config,machine):
 e=MACHINES[machine]
 assert config['printer_model']==e['model'] and config['curr_bed_type']==e['bed']
 assert config['filament_ids']==[e['material']] and config['filament_type']==['PLA'] and config['nozzle_diameter']==['0.4']
 assert config['printable_area']==['0x0','256x0','256x256','0x256'],'This packer targets the verified 256 mm beds'

def default_profile_refs():
 rows={}
 for c in COLORS:
  for machine in MACHINES:
   p=OLD/f'{c}_{machine}_editable.3mf'
   if not p.exists() and c=='White' and machine=='Bambu':p=OLD.parent/'delivery-V1.3-Incremental/White/White_V1.3_editable.3mf'
   rows[c+'_'+machine]=ref(p)
 return rows

PLATE_CENTRE_TOL_MM=1.0   # model footprint centre to bed centre (128, 128), both axes

def arrange(parts,margin=30,gap=12):
 # Shared placement for both machines; central 196x196 square avoids both
 # manufacturer's corner exclusion regions. Native support footprint follows.
 placements=[];x=y=margin;row_h=0
 for row in parts:
  b=np.array(row['bounds']);w,h=b[1,:2]-b[0,:2]
  assert max(w,h)<256-2*margin and b[1,2]<250,'Part exceeds common printer build volume'
  for i in range(row['quantity']):
   if x+w>256-margin:x=margin;y+=row_h+gap;row_h=0
   assert y+h<=256-margin,'Single color plate full; supply explicit placements in release manifest'
   placements.append(dict(key=row['key'],copy=i+1,translation=[float(x-b[0,0]),float(y-b[0,1]),0.]))
   x+=w+gap;row_h=max(row_h,h)
 # Center complete model footprint, not a long off-center row.
 if placements:
  bykey={r['key']:r for r in parts}
  boxes=[np.array(bykey[r['key']]['bounds'])+r['translation'] for r in placements]
  extent=np.array([np.min([b[0]for b in boxes],axis=0),np.max([b[1]for b in boxes],axis=0)])
  shift=128-(extent[0,:2]+extent[1,:2])/2
  for row in placements:row['translation'][:2]=(np.array(row['translation'][:2])+shift).tolist()
 return placements

def matrix_text(T):return ' '.join(format(float(x),'.15g')for x in T[:3,:].T.flatten())
def mesh_object(resources,oid,name,mesh):
 obj=ET.SubElement(resources,Q+'object',{'id':str(oid),'type':'model','name':name});el=ET.SubElement(obj,Q+'mesh');ve=ET.SubElement(el,Q+'vertices');fa=ET.SubElement(el,Q+'triangles')
 for v in mesh.vertices:ET.SubElement(ve,Q+'vertex',{k:format(float(x),'.15g')for k,x in zip(('x','y','z'),v)})
 for f in mesh.faces:ET.SubElement(fa,Q+'triangle',{k:str(int(x))for k,x in zip(('v1','v2','v3'),f)})
 return obj

def bambu_self_index(cfg):
 """Bambu Studio 02.08 GUI throws "Invalid configuration file" unless filament_self_index has one entry per
 filament_extruder_variant row (PresetBundle::load_config_file_config, v02.08.02.61 PresetBundle.cpp L3794-3810). The donor
 projects lack the key, so the CLI fills its schema default [1] while keeping two variant rows (Direct Drive Standard and
 High Flow). A GUI-saved single-filament project sizes it to the rows, all 1 (PresetBundle.cpp L3358-3360)."""
 nfil=len(cfg.get('filament_colour',[]));rows=cfg.get('filament_extruder_variant',[])
 assert nfil==1,f'bambu_self_index handles one filament per plate; got {nfil}'
 cfg=dict(cfg);cfg['filament_self_index']=['1']*len(rows);return cfg

def check_bambu_archive(path):
 """A delivered X1C .gcode.3mf must open in the Bambu Studio GUI on Preview with its stored G-code: a consistent
 filament_self_index (no "Invalid configuration file") and no model objects, because the GUI reuses plate_N.gcode only
 when model.objects is empty (Plater.cpp L9579-9614); the CLI writes no objects only with --min-save (BambuStudio.cpp L8487)."""
 with zipfile.ZipFile(path)as z:
  cfg=json.loads(z.read('Metadata/project_settings.config'));names=z.namelist();mdl=z.read('3D/3dmodel.model').decode(errors='replace')
 assert len(cfg.get('filament_self_index',[]))==len(cfg.get('filament_extruder_variant',[])),f'{path}: filament_self_index does not match filament_extruder_variant (GUI "Invalid configuration file")'
 assert '<object' not in mdl and not any(n.startswith('3D/Objects/')for n in names),f'{path}: model objects in the sliced archive (GUI drops the stored slice and opens Prepare)'
 assert any(n.startswith('Metadata/plate_')and n.endswith('.gcode')for n in names),f'{path}: no stored G-code'

import re as _re
OTHER_MATERIALS=_re.compile(r'\b(PETG|ABS|ASA|TPU|PC|nylon|PA12)\b',_re.I)
def check_material_words(manifest_path):
 """Every MicroDuck plate is sliced with the PLA profiles (check_profile asserts filament_type == PLA). A manifest that names another
 material in its prose is wrong (2026-10-25: 'white PETG out' was written from an assumption and spread to the guide and DECISIONS)."""
 txt=Path(manifest_path).read_text();hits=sorted({m.group(0) for m in OTHER_MATERIALS.finditer(txt)})
 assert not hits,f'{manifest_path} names a material other than PLA: {hits}; the plates are PLA (check_profile)'

def prepare(manifest_path,out):
 out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
 check_material_words(manifest_path)
 m,receipt=source_receipt(manifest_path);sources={r['key']:r for r in m['parts']};rows=[]
 for color in COLORS:
  parts=[r for r in receipt['parts']if r['color']==color]
  if not parts:continue
  placements=arrange(parts) if 'placements' not in m else m['placements'][color]
  assert sorted((r['key'],r['copy'])for r in placements)==sorted((r['key'],i+1)for r in parts for i in range(r['quantity']))
  for machine in MACHINES:
   donor=bound(m['profiles'][color+'_'+machine])
   with zipfile.ZipFile(donor)as z:cfg=json.loads(z.read('Metadata/project_settings.config'));types=z.read('[Content_Types].xml');rels=z.read('_rels/.rels')
   check_profile(cfg,machine)
   cfg['filename_format']='zhengxia_'+m['version']+'_'+color+'_'+machine+'.gcode'
   cfg.update(m.get('processOverrides',{}).get(color+'_'+machine,{}));check_profile(cfg,machine)
   if machine=='Bambu':cfg=bambu_self_index(cfg)
   model=ET.Element(Q+'model',{'unit':'millimeter','{http://www.w3.org/XML/1998/namespace}lang':'en-US'})
   app=ET.SubElement(model,Q+'metadata',{'name':'Application'});app.text='BambuStudio-02.08.02.61'if machine=='Bambu'else'OrcaSlicer-2.4.2'
   resources=ET.SubElement(model,Q+'resources');build=ET.SubElement(model,Q+'build');settings=ET.Element('config');plate=ET.SubElement(settings,'plate')
   for k,v in [('plater_id','1'),('plater_name',color),('locked','false'),('bed_type',cfg['curr_bed_type'])]:ET.SubElement(plate,'metadata',{'key':k,'value':v})
   oid=1;boxes=[];modrefs=[]
   for placement in placements:
    key=placement['key'];s=sources[key];r=next(r for r in parts if r['key']==key);name=key+'_'+str(placement['copy']);T=np.array(r['sourceToPrint']);T[2,3]+=r['dropToBedMm'];mesh=load_mesh(s['path'],T);parent_id=oid;oid+=1
    modifiers=[v for v in s.get('supportBlockers',[])if machine in v.get('machines',list(MACHINES))]
    if modifiers:
     parent=ET.SubElement(resources,Q+'object',{'id':str(parent_id),'type':'model','name':name});components=ET.SubElement(parent,Q+'components');mid=oid;oid+=1;mesh_object(resources,mid,name,mesh);ET.SubElement(components,Q+'component',{'objectid':str(mid)})
    else:mid=parent_id;mesh_object(resources,mid,name,mesh)
    st=ET.SubElement(settings,'object',{'id':str(parent_id)})
    for k,v in [('name',name),('extruder','1')]+sorted(r['objectSettings'].items()):ET.SubElement(st,'metadata',{'key':k,'value':v})
    part=ET.SubElement(st,'part',{'id':str(mid),'subtype':'normal_part'});ET.SubElement(part,'metadata',{'key':'name','value':name})
    for mod in modifiers:
     mp=bound(mod);modrefs.append(ref(mp));mm=load_mesh(mp,T);idx=oid;oid+=1;mesh_object(resources,idx,name+'_support_blocker',mm);ET.SubElement(components,Q+'component',{'objectid':str(idx)})
     part=ET.SubElement(st,'part',{'id':str(idx),'subtype':'support_blocker'});ET.SubElement(part,'metadata',{'key':'name','value':name+'_support_blocker'})
    translation=np.array(placement['translation'],dtype=float);assert translation.shape==(3,) and abs(translation[2])<1e-9
    b=mesh.bounds+translation;assert np.min(b[0,:2])>=18 and np.max(b[1,:2])<=238 and b[1,2]<250 and abs(b[0,2])<1e-6,'Bed/exclusion violation'
    for oldb in boxes:
     assert b[0,0]>=oldb[1,0]+4 or oldb[0,0]>=b[1,0]+4 or b[0,1]>=oldb[1,1]+4 or oldb[0,1]>=b[1,1]+4,'Model separation under 4 mm'
    boxes.append(b);B=np.eye(4);B[:3,3]=translation;ET.SubElement(build,Q+'item',{'objectid':str(parent_id),'transform':matrix_text(B),'printable':'1'})
    ins=ET.SubElement(plate,'model_instance')
    for k,v in [('object_id',str(parent_id)),('instance_id','0'),('identify_id',str(2000+parent_id))]:ET.SubElement(ins,'metadata',{'key':k,'value':v})
   # Centred plate (Shawn 2026-10-15: the printed V1.18 plate sat 38 mm left and 63 mm forward of centre). arrange() centres
   # its own layout; explicit manifest placements skipped that until this check. The whole model footprint must be centred.
   lo=np.min([b[0,:2]for b in boxes],axis=0);hi=np.max([b[1,:2]for b in boxes],axis=0);centre_off=(lo+hi)/2-128
   assert np.all(np.abs(centre_off)<=PLATE_CENTRE_TOL_MM),f'{color} plate footprint not centred: centre offset {centre_off.round(2).tolist()} mm (limit {PLATE_CENTRE_TOL_MM}); shift every placement by {(-centre_off).round(2).tolist()}'
   path=out/f'{color}_{machine}_editable.3mf'
   with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED)as z:
    for n,b in {'[Content_Types].xml':types,'_rels/.rels':rels,'3D/3dmodel.model':ET.tostring(model,encoding='utf-8',xml_declaration=True),'Metadata/model_settings.config':ET.tostring(settings,encoding='utf-8',xml_declaration=True),'Metadata/project_settings.config':json.dumps(cfg,indent=2).encode()}.items():z.writestr(n,b)
   rows.append(dict(color=color,machine=machine,project=ref(path),profile=ref(donor),supportBlockers=modrefs,placements=placements,modelBounds=[b.tolist()for b in boxes],plateCentreOffsetMm=centre_off.round(3).tolist()))
 receipt.update(status='PREPARED_NOT_SLICED',rows=rows);write(out/'preparation.json',receipt);return receipt

def slice_one(preparation_path,color,machine):
 assert __import__('os').environ.get('MICRODUCK_BOUNDED_CAD'),'Run through run_bounded_cad.py exclusive resource guard'
 p=Path(preparation_path).resolve();receipt=read(p);bound(receipt['manifest']);bound(receipt['pipeline'])
 row=next(r for r in receipt['rows']if r['color']==color and r['machine']==machine);project=bound(row['project']);bound(row['profile'])
 for s in receipt['parts']:bound(s['source']);bound(s['incrementalEvidence'])
 for r in row['supportBlockers']:bound(r)
 out=p.parent/'native'/f'{color}_{machine}';out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(project)as z:cfg=json.loads(z.read('Metadata/project_settings.config'))
 check_profile(cfg,machine);cmd=[MACHINES[machine]['exe'],'--allow-newer-file','--arrange','0','--orient','0','--slice','0']
 extra=[]
 if machine=='Elegoo':
  fp=HERE.parent/'profiles/elegoo-filament.json'  # cmu/ copy: the Elegoo filament profile ships beside the script;extra=[ref(fp)];cmd+=['--load-filaments',str(fp)]
 if machine=='Bambu':cmd+=['--min-save']   # GUI-style sliced file: no model objects, so Bambu Studio opens it on Preview
 cmd+=['--export-3mf','sliced.3mf','--outputdir',str(out),str(project)]
 before=set((Path.home()/'Library/Logs/DiagnosticReports').glob(('BambuStudio'if machine=='Bambu'else'OrcaSlicer')+'*.ips'))
 write(out/'command.json',dict(argv=cmd,preparation=ref(p),project=ref(project),extraProfiles=extra))
 with(out/'cli.log').open('w')as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=170)
 assert result.returncode==0,f'Native failure {result.returncode}: {out/"cli.log"}'
 after=set((Path.home()/'Library/Logs/DiagnosticReports').glob(('BambuStudio'if machine=='Bambu'else'OrcaSlicer')+'*.ips'));assert not(after-before),'New native crash report'
 archive=out/'sliced.3mf'
 with zipfile.ZipFile(archive)as z:
  members=[n for n in z.namelist()if n.endswith('.gcode')];assert len(members)==1;g=z.read(members[0]);check_profile(json.loads(z.read('Metadata/project_settings.config')),machine)
 assert b'G1 'in g
 if machine=='Bambu':check_bambu_archive(archive)
 write(out/'result.json',dict(passed=True,preparation=ref(p),project=ref(project),archive=ref(archive),gcodeSha256=digest(g),command=ref(out/'command.json')))
 return archive

def native_payload(p):
 with zipfile.ZipFile(p)as z:
  ns=[n for n in z.namelist()if n.endswith('.gcode')];assert len(ns)==1
  return z.read(ns[0]),json.loads(z.read('Metadata/project_settings.config')),{n:z.read(n)for n in z.namelist()if n.endswith('.png')}
def time_seconds(code,machine):
 pattern=r'total estimated time:\s*([^\r\n]+)'if machine=='Bambu'else r'estimated printing time \(normal mode\)\s*=\s*([^\r\n]+)'
 match=re.search(pattern,code.decode());assert match,'Native ETA missing'
 seconds=sum(int(n)*{'h':3600,'m':60,'s':1}[u]for n,u in re.findall(r'(\d+)\s*([hms])',match.group(1)));assert seconds>0
 return seconds

# Centauri Carbon firmware (github.com/elegooofficial/CentauriCarbon, firmware/core/klippy/gcode_preview.c) parses the
# preview, ETA and filament only if 'OrcaSlicer'/'ElegooSlicer' occurs in the first 30 lines, and stops scanning for a
# preview at the first G/M command. Elegoo's own sample files carry a 144x144 PNG right after HEADER_BLOCK_END.
CC_SLICER_LINES=30
def cc_preview(g,png):
 from PIL import Image
 size=re.search(rb'^; thumbnails = (\d+)x(\d+)/PNG',g,re.M);w,h=(int(size[1]),int(size[2]))if size else(144,144)
 buf=io.BytesIO();Image.open(io.BytesIO(png)).convert('RGBA').resize((w,h),Image.LANCZOS).save(buf,'PNG',optimize=True);enc=base64.b64encode(buf.getvalue()).decode()
 block=(f'; THUMBNAIL_BLOCK_START\n\n;\n; thumbnail begin {w}x{h} {len(enc)}\n'+''.join('; '+enc[i:i+78]+'\n'for i in range(0,len(enc),78))+'; thumbnail end\n; THUMBNAIL_BLOCK_END\n\n').encode()
 k=b'; HEADER_BLOCK_END\n\n';i=g.index(k)+len(k);payload=g[:i]+block+g[i:];check_cc_preview(payload,g);return payload
def check_cc_preview(payload,g):
 """Replay the Centauri Carbon preview parser and prove the only change to the native command stream is the preview block."""
 from PIL import Image,ImageStat
 lines=payload.split(b'\n')
 assert any(s in l for l in lines[:CC_SLICER_LINES]for s in(b'OrcaSlicer',b'ElegooSlicer')),'Centauri Carbon cannot identify the slicer in the first 30 lines; it will show no preview'
 n=next((i for i,l in enumerate(lines)if l.startswith(b'; thumbnail begin ')or l[:1]in(b'G',b'M',b'g',b'm')),None)
 assert n is not None and lines[n].startswith(b'; thumbnail begin '),'No preview before the first G-code command'
 dims,size=lines[n].split()[3:5];w,h=map(int,dims.split(b'x'))
 end=next(i for i in range(n+1,len(lines))if lines[i].startswith(b'; thumbnail end'));raw=b''.join(l[2:]for l in lines[n+1:end])
 assert len(raw)==int(size),'Preview length does not match its header'
 im=Image.open(io.BytesIO(base64.b64decode(raw)));assert im.size==(w,h)and max(ImageStat.Stat(im.convert('RGB')).stddev)>2,'Blank or mis-sized preview'
 if payload!=g:
  s=payload.index(b'; THUMBNAIL_BLOCK_START');e=payload.index(b'; THUMBNAIL_BLOCK_END\n\n')+len(b'; THUMBNAIL_BLOCK_END\n\n');assert payload[:s]+payload[e:]==g,'Command stream changed'
 return dict(width=w,height=h,slicerLine=next(i for i,l in enumerate(lines,1)if b'OrcaSlicer'in l or b'ElegooSlicer'in l),previewLine=n+1)

def first_layer(g,project,machine):
 import shapely
 from shapely.geometry import LineString,box
 from shapely.ops import unary_union
 with zipfile.ZipFile(project)as z:
  cfg=json.loads(z.read('Metadata/project_settings.config'));settings=ET.fromstring(z.read('Metadata/model_settings.config'))
 height=float(cfg['initial_layer_print_height']);names={o.get('id'):o.find("metadata[@key='name']").get('value')for o in settings.findall('object')}
 labels={x.find("metadata[@key='identify_id']").get('value'):names[x.find("metadata[@key='object_id']").get('value')]for x in settings.findall('plate/model_instance')}
 offset=list(map(float,cfg.get('extruder_offset',['0x0'])[0].split('x'))) if machine=='Bambu' else [0.,0.]
 xyz=[0.,0.,0.];width=.4;kind='Custom';cur=None;relative=True;e=0.;paths={};allbounds={};maxwidth=.4;loose=[]
 for line in g.decode().splitlines():
  if line.startswith('; start printing object, unique label id:'):cur=labels[line.rsplit(':',1)[1].strip()]
  elif line.startswith('; stop printing object, unique label id:'):cur=None
  elif line.startswith('; OBJECT_ID:'):cur=labels.get(line.split(':',1)[1].strip())
  elif line.startswith('EXCLUDE_OBJECT_START'):cur=re.search(r'NAME=(\S+)',line)[1].split('_id_')[0]
  elif line.startswith('EXCLUDE_OBJECT_END'):cur=None
  elif line.startswith(('; FEATURE:',';TYPE:')):kind=line.split(':',1)[1].strip()
  elif line.startswith(('; LINE_WIDTH:',';WIDTH:')):width=float(line.split(':',1)[1]);maxwidth=max(maxwidth,width)
  if line.startswith('M82'):relative=False
  if line.startswith('M83'):relative=True
  if line.startswith('G92 '):
   match=re.search(r'E([-+\d.]+)',line)
   if match:e=float(match[1])
  if not re.match(r'^G[0123] ',line):continue
  v={k:float(t)for k,t in re.findall(r'([XYZEIJ])([-+]?\d*\.?\d+)',line.split(';')[0])}
  end=[v.get(k,xyz[i])for i,k in enumerate('XYZ')];de=v.get('E',0) if relative else v.get('E',e)-e
  if 'E'in v:e=v['E']if not relative else e+v['E']
  # OrcaSlicer prints each object's brim just before EXCLUDE_OBJECT_START, outside the label; Bambu prints it inside.
  if (cur or 'brim'in kind.lower()) and de>0 and kind!='Custom' and (end[:2]!=xyz[:2] or line.startswith(('G2 ','G3 '))):
   pts=[xyz[:2],end[:2]];boundspts=pts;onfirst=abs(end[2]-height)<1e-5
   if line.startswith(('G2 ','G3 ')):
    cx=xyz[0]+v.get('I',0);cy=xyz[1]+v.get('J',0);rad=math.hypot(xyz[0]-cx,xyz[1]-cy);aa=math.atan2(xyz[1]-cy,xyz[0]-cx);bb=math.atan2(end[1]-cy,end[0]-cx);sign=1 if line.startswith('G3 ')else -1;span=((bb-aa)*sign)%(2*math.pi)
    if span<1e-9:span=2*math.pi
    boundspts=pts+[(cx+rad*math.cos(t),cy+rad*math.sin(t))for t in (0,math.pi/2,math.pi,3*math.pi/2)if ((t-aa)*sign)%(2*math.pi)<=span+1e-9]
    if onfirst:
     n=max(8,math.ceil(span*rad/.05));pts=[(cx+rad*math.cos(aa+sign*span*i/n),cy+rad*math.sin(aa+sign*span*i/n))for i in range(n+1)]
   if not cur:
    # Unlabelled brim: owner assigned below by nearest first-layer model; its bounds join that owner's footprint.
    loose.append((LineString([(x+offset[0],y+offset[1])for x,y in pts]).buffer(width/2),[(x+offset[0],y+offset[1])for x,y in boundspts],onfirst,kind))
    xyz=end;continue
   # A conservative rectangle needs only running extrema. Keeping every
   # interpolated arc point in Python grew to gigabytes on a normal head slice.
   bb=allbounds.setdefault(cur,[math.inf,math.inf,-math.inf,-math.inf])
   for x,y in boundspts:
    x+=offset[0];y+=offset[1];bb[0]=min(bb[0],x);bb[1]=min(bb[1],y);bb[2]=max(bb[2],x);bb[3]=max(bb[3],y)
   if onfirst:
    pts=[(x+offset[0],y+offset[1])for x,y in pts];paths.setdefault(cur,{}).setdefault(kind,[]).append(LineString(pts).buffer(width/2))
  xyz=end
 models={name:unary_union([shape for kind,ss in kinds.items()if not any(t in kind.lower()for t in ('support','brim','skirt'))for shape in ss])for name,kinds in paths.items()}
 models={k:v for k,v in models.items()if not v.is_empty}
 assert models,'Missing model first-layer contact'
 loose_owner={}
 if loose:
  owners=list(models);tree=shapely.STRtree([models[k]for k in owners])
  nearest=tree.query_nearest([s for s,_,_,_ in loose],all_matches=False)
  for i,j in zip(*nearest):
   s,bpts,onfirst,kind=loose[int(i)];name=owners[int(j)];loose_owner[int(i)]=name
   bb=allbounds.setdefault(name,[math.inf,math.inf,-math.inf,-math.inf])
   for x,y in bpts:bb[0]=min(bb[0],x);bb[1]=min(bb[1],y);bb[2]=max(bb[2],x);bb[3]=max(bb[3],y)
   if onfirst:paths[name].setdefault(kind,[]).append(s)
 rows=[]
 for name,kinds in paths.items():
  model=unary_union([shape for kind,ss in kinds.items()if not any(t in kind.lower()for t in ('support','brim','skirt'))for shape in ss]);support=unary_union([shape for kind,ss in kinds.items()if 'support'in kind.lower()for shape in ss])
  if model.is_empty:continue
  brim=unary_union([shape for kind,ss in kinds.items()if 'brim'in kind.lower()for shape in ss])
  if brim.is_empty:inside=reach=reach_own=0.
  else:
   inside=brim.intersection(model.convex_hull).area
   ring=shapely.points(shapely.get_coordinates(brim.boundary));reach=float(shapely.distance(ring,model).max())
   dm=shapely.distance(ring,model);own=dm<=shapely.distance(ring,support)if not support.is_empty else dm==dm
   reach_own=float(dm[own].max())if own.any()else 0.   # the part's own brim: outline points nearer the part than its support
  rows.append(dict(name=name,model_area_mm2=model.area,modelIslands=len(model.geoms)if hasattr(model,'geoms')else 1,support_area_mm2=support.area,firstLayerMm=height,
                   brim_area_mm2=brim.area,brimInsideOwnHullMm2=inside,brimReachMm=reach,brimOwnReachMm=reach_own,brimUnlabelledSegments=sum(1 for i,n in loose_owner.items()if n==name and loose[i][2])))
 assert rows and all(r['model_area_mm2']>0 for r in rows),'Missing model first-layer contact'
 rad=maxwidth/2+.01;hulls={k:box(*v).buffer(rad)for k,v in allbounds.items()};bed=box(0,0,256,256);exclusion=box(0,0,18,28)if machine=='Bambu'else box(246,0,256,20)
 assert hulls and all(bed.covers(h)and not h.intersects(exclusion)for h in hulls.values()),'Native extrusion outside printable bed'
 # Conservative rectangle overlap requires a separate exact spacing review.
 pairs=[dict(a=a,b=b,conservativeGapMm=hulls[a].distance(hulls[b]))for i,a in enumerate(hulls)for b in list(hulls)[i+1:]]
 return rows,dict(allLayerModelSupportBrimBounds={k:list(h.bounds)for k,h in hulls.items()},insideBed=True,pairs=pairs,extruderOffsetAppliedMm=offset)

def brim_review(rows,parts,machine=None):
 """Per-object brim against the per-part plan (prep parts with objectSettings). rows: first_layer() rows of one plate.
 Fails on: brim on a no_brim part; planned brim missing; planned brim reach off brim_object_gap + brim_width by more than
 BRIM_REACH_TOL_MM (a slicer that ignored the per-object width); brim inside the part's own convex first-layer outline
 above BRIM_INSIDE_HULL_MAX_MM2 (an open opening such as the C-shaped speaker retainer filled with brim)."""
 plan={f'{p["key"]}_{i+1}':p for p in parts for i in range(p['quantity'])};seen=set();out={};ok=True
 for r in rows:
  name=r['name'];assert name in plan,f'{name}: object not in the preparation parts';seen.add(name);s=plan[name].get('objectSettings')
  assert s is not None,f'{name}: no per-object brim plan (preparation from pipeline v1?)'
  planned=planned_brim(s);area=round(r['brim_area_mm2'],1);inside=round(r['brimInsideOwnHullMm2'],1);reach=round(r['brimReachMm'],2);fails=[]
  inside_max=max(BRIM_INSIDE_HULL_MAX_MM2,float(plan[name].get('brimInsideHullAllowanceMm2',0)))   # a manifest row's hull_allowance
  if planned:
   want=float(s['brim_object_gap'])+float(s['brim_width'])
   if area<=0:fails.append('planned brim missing')
   else:
    if 'brimAlongSupportReason'in plan[name]:   # support_brim(): the brim also runs round the part's own first-layer support
     own=round(r['brimOwnReachMm'],2)
     if own<=0:fails.append('no brim outline nearer the part than its support')
     elif own>want+BRIM_REACH_TOL_MM:fails.append(f'own brim reach {own} mm (outline nearer the part than its support), planned {want:g} mm (gap + width)')
    elif abs(reach-want)>BRIM_REACH_TOL_MM:fails.append(f'brim reach {reach} mm, planned {want:g} mm (gap + width)')
    if inside>inside_max:fails.append(f'{inside} mm2 of brim inside the part\'s own convex outline')
  elif area>0:fails.append(f'unplanned brim {area} mm2'+(f', {inside} mm2 of it inside the part\'s own convex outline'if inside>inside_max else''))
  out[name]=dict(brimType=s.get('brim_type'),planned=planned,brimAreaMm2=area,modelAreaMm2=round(r['model_area_mm2'],1),brimInsideOwnHullMm2=inside,insideOwnHullMaxMm2=inside_max,brimReachMm=reach,
                 plannedReachMm=float(s['brim_object_gap'])+float(s['brim_width'])if planned else None,failures=fails,passed=not fails,
                 **({'allowanceReason':plan[name]['brimInsideHullAllowanceReason']}if 'brimInsideHullAllowanceMm2'in plan[name]else{}),
                 **({'brimOwnReachMm':round(r['brimOwnReachMm'],2),'brimAlongSupportReason':plan[name]['brimAlongSupportReason']}
                    if 'brimAlongSupportReason'in plan[name]else{}))
  ok&=not fails
 missing=sorted(set(plan)-seen)
 if missing:ok=False
 return dict(passed=bool(ok),machine=machine,objects=out,missingObjects=missing,
             limits=dict(insideOwnHullMaxMm2=BRIM_INSIDE_HULL_MAX_MM2,reachToleranceMm=BRIM_REACH_TOL_MM))

def package(preparation_path,review_path=None):
 from PIL import Image,ImageStat
 p=Path(preparation_path).resolve();prep=read(p);manifest=bound(prep['manifest']);m,check=source_receipt(manifest);assert check['parts']==prep['parts'];bound(prep['pipeline'])
 out=p.parent/'delivery';out.mkdir(exist_ok=True);assert not list(out.iterdir()),'Use a new output directory; never overwrite a delivery'
 files=[];checks=[]
 for row in prep['rows']:
  color,machine=row['color'],row['machine'];project=bound(row['project']);resultp=p.parent/'native'/f'{color}_{machine}'/'result.json';result=read(resultp)
  assert result['passed'] and result['preparation']==ref(p) and result['project']==ref(project)
  command=read(bound(result['command']));assert command['preparation']==ref(p) and command['project']==ref(project)
  for extra in command['extraProfiles']:bound(extra)
  source=bound(result['archive']);g,cfg,imgs=native_payload(source);assert digest(g)==result['gcodeSha256'];check_profile(cfg,machine)
  actualfirst,footprint=first_layer(g,project,machine);assert len(actualfirst)==sum(r['quantity']for r in prep['parts']if r['color']==color),'Slicer lost or added model objects'
  brims=brim_review(actualfirst,[r for r in prep['parts']if r['color']==color],machine)
  assert brims['passed'],f'Brim review failed on {color} {machine}: '+json.dumps({k:v['failures']for k,v in brims['objects'].items()if v['failures']})
  seconds=time_seconds(g,machine);minutes=math.ceil(seconds/60);label=(f'{minutes//60}h'if minutes>=60 else'')+f'{minutes%60}m'
  name=f'zhengxia_{m["version"]}_{color}_{MACHINES[machine]["slug"]}_{label}'+('.gcode.3mf'if machine=='Bambu'else'.gcode');dest=out/name
  previewSource=source
  if machine=='Elegoo' and 'Metadata/plate_1.png'not in imgs:
   # Same-layout Bambu native image is explicitly identified; no fabricated GUI claim.
   peer=next(x for x in prep['rows']if x['color']==color and x['machine']=='Bambu');assert peer['placements']==row['placements']
   pr=read(p.parent/'native'/f'{color}_Bambu/result.json');assert pr['preparation']==ref(p);previewSource=bound(pr['archive']);_,_,imgs=native_payload(previewSource)
  png=imgs['Metadata/plate_1.png'];im=Image.open(io.BytesIO(png)).convert('RGB');assert min(im.size)>=100 and max(ImageStat.Stat(im).stddev)>2,'Blank thumbnail'
  if machine=='Bambu':check_bambu_archive(source);shutil.copyfile(source,dest)
  else:
   payload=g if b'thumbnail begin' in g else cc_preview(g,png);check_cc_preview(payload,g)
   dest.write_bytes(payload)
  files.append(dict(color=color,machine=MACHINES[machine]['label'],path=str(dest),filename=name,estimatedSeconds=seconds,estimatedTime=label,sha256=sha(dest),plate=MACHINES[machine]['bed']))
  checks.append(dict(color=color,machine=machine,result=ref(resultp),nativeArchive=ref(source),nativeCommandStreamSha256=digest(g),nativeFirstLayer=actualfirst,nativeFootprint=footprint,brimReview=brims,thumbnailSource=ref(previewSource),thumbnailSha256=digest(png)))
 colors={r['color']for r in m['parts']};assert len(files)==2*len(colors)
 status='FIXTURE_ONLY_NOT_RELEASED'if m['releaseKind']=='fixture'else'PREPARED_NATIVE_REVIEW_PENDING'
 # Final reviewer binds the actual sliced footprints/functional holes. It is a
 # receipt from the normal native review, not a fresh whole-machine CAD gate.
 review=ref(review_path) if review_path else None
 if review and m['releaseKind']=='incremental':
  rv=read(bound(review));assert rv['passed'] and rv['preparationSha256']==sha(p)
  assert rv['nativeArchives']=={str(bound(read(p.parent/'native'/f'{r["color"]}_{r["machine"]}/result.json')['archive'])):sha(bound(read(p.parent/'native'/f'{r["color"]}_{r["machine"]}/result.json')['archive']))for r in prep['rows']}
  status='RELEASED'
 audit=dict(version=m['version'],status=status,baselinePrintedStockInstances=89,partTypes=check['partTypes'],printQuantity=check['printQuantity'],reprintUnchangedStockQuantity=check['reprintUnchangedStockQuantity'],manifest=ref(manifest),parts=prep['parts'])
 write(p.parent/'incremental-audit.json',audit);write(p.parent/'native-delivery-checks.json',dict(passed=True,preparation=ref(p),checks=checks,claimBoundary='Native toolpaths only; no printer execution or physical fit. Per-object brim checked against the plan; final footprint/hole review is separate.'))
 index=dict(version=m['version'],status=status,distinctPartTypes=check['partTypes'],printedQuantity=check['printQuantity'],files=files,manifest=ref(manifest),preparation=ref(p),incrementalAudit=ref(p.parent/'incremental-audit.json'),nativeChecks=ref(p.parent/'native-delivery-checks.json'))
 if review:index['nativeReview']=review
 write(p.parent/'delivery-index.json',index)
 zpath=p.parent/(f'zhengxia_{m["version"]}_'+('TEST_ONLY_'if m['releaseKind']=='fixture'else'')+'SD-files.zip')
 with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED)as z:
  for row in files:z.write(row['path'],row['filename'])
 with zipfile.ZipFile(zpath)as z:
  assert set(z.namelist())=={r['filename']for r in files}
  assert all(digest(z.read(r['filename']))==r['sha256']for r in files)
 write(p.parent/'package-receipt.json',dict(status=status,index=ref(p.parent/'delivery-index.json'),zip=ref(zpath),exactFiles=len(files),containsSourceOrInternalFiles=False));return index

def probe(out):
 rows=[]
 for machine,e in MACHINES.items():
  exe=Path(e['exe']);assert exe.is_file() and __import__('os').access(exe,__import__('os').X_OK)
  r=subprocess.run([str(exe),'--help'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20);s=r.stdout.decode(errors='replace')
  assert all(x in s for x in ('--slice','--export-3mf','--arrange','--orient'))
  rows.append(dict(machine=machine,executable=ref(exe),helpExitCode=r.returncode,helpSha256=digest(r.stdout),supportedFlags=True))
 profiles=default_profile_refs()
 for k,v in profiles.items():
  with zipfile.ZipFile(bound(v))as z:check_profile(json.loads(z.read('Metadata/project_settings.config')),k.split('_')[-1])
 write(out,dict(passed=True,tools=rows,profiles=profiles,scope='Executable and profile schema probe; native fixture is separate'))

if __name__=='__main__':
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='action',required=True)
 a=sub.add_parser('prepare');a.add_argument('manifest');a.add_argument('output')
 a=sub.add_parser('slice');a.add_argument('preparation');a.add_argument('color');a.add_argument('machine',choices=MACHINES)
 a=sub.add_parser('package');a.add_argument('preparation');a.add_argument('--native-review')
 a=sub.add_parser('probe');a.add_argument('output')
 args=ap.parse_args()
 if args.action=='prepare':v=prepare(args.manifest,args.output);print('Prepared',len(v['rows']),'native projects; no slicing')
 elif args.action=='slice':print(slice_one(args.preparation,args.color,args.machine))
 elif args.action=='package':v=package(args.preparation,args.native_review);print(v['status'],len(v['files']),'files')
 else:probe(args.output);print('Profiles and executables available')
