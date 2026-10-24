"""Minimum wall-thickness check for print parts. release_pipeline_v2.source_receipt runs it on every part row, so prepare
refuses a part with a thin wall and preparation.json records the result per part under `wallCheck`.

A wall must be at least MIN_WALL_MM thick: three 0.4 mm extrusion lines. Metric, on the print-frame mesh:
1. Sample the surface evenly, about one point per SAMPLE_AREA_MM2 (at most MAX_SAMPLES points), with face normals.
2. Cast a ray from each point, offset RAY_OFFSET_MM inside, along the inward normal; raw thickness is the distance to the
   first hit, capped at RAY_MAX_MM.
3. Smoothed thickness is the median raw thickness of the points within NEIGHBOUR_MM. A concave corner gives short rays on
   a narrow strip only, so the median removes it; a real thin wall is thin across its whole neighbourhood.
4. Points with smoothed thickness below MIN_WALL_MM - MEASURE_TOL_MM are thin. Single linkage at NEIGHBOUR_MM groups them
   into regions; region area is the point count times the area per sample.
5. A region of at least MIN_REGION_MM2 fails, unless a `thinWallAllowances` entry {bbox, minMm, reason} on the part row
   contains the region bbox (print frame, mm) and the region's minimum smoothed thickness is at least minMm.

The metric measures material thickness. A thick-backed protrusion (a 0.4 mm proud rib on a solid face) reads thick.

    /tmp/microduck-cad-venv/bin/python wall_check.py PART.stl [PART.stl ...]

checks STLs in their own frame and prints the regions of at least MIN_REGION_MM2.
"""
import json,sys,time
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MIN_WALL_MM=1.2         # three 0.4 mm lines (Shawn 2026-10-19)
MIN_REGION_MM2=2.0      # thin surface area that fails a part
NEIGHBOUR_MM=1.0        # smoothing radius and single-linkage distance
MEASURE_TOL_MM=.01      # a wall modelled at exactly MIN_WALL_MM measures 1.19999 in float
SAMPLE_AREA_MM2=.05     # surface area per sample point
MAX_SAMPLES=60000
RAY_OFFSET_MM=1e-3
RAY_MAX_MM=4.0          # raw thickness cap; must stay >= 2 x MIN_WALL_MM (see measure)
PAIR_CHUNK=200000       # ray-triangle pairs per vectorised batch (bounds memory)
SEED=0                  # fixed, so a recomputed receipt matches the prepared one
BBOX_TOL_MM=1e-6

def first_hit(mesh,origins,dirs,max_mm=RAY_MAX_MM):
 """Distance along each unit ray to the first triangle closer than max_mm, else max_mm. Exact Moller-Trumbore on the
 triangles whose boxes meet the ray segment's box (the mesh's rtree). trimesh's unbounded ray_triangle intersector took
 3.2 GB and 21 s for 60k rays on the V1.22 foot (embree is not installed); this gives the same first hits in 0.3 GB."""
 out=np.full(len(origins),float(max_mm))
 ends=origins+dirs*max_mm;ids,counts=mesh.triangles_tree.intersection_v(np.minimum(origins,ends),np.maximum(origins,ends))
 ids=np.asarray(ids,dtype=np.int64);ray=np.repeat(np.arange(len(origins)),np.asarray(counts,dtype=np.int64));tris=mesh.triangles
 for s in range(0,len(ids),PAIR_CHUNK):
  r=ray[s:s+PAIR_CHUNK];t=tris[ids[s:s+PAIR_CHUNK]];o=origins[r];d=dirs[r]
  e1=t[:,1]-t[:,0];e2=t[:,2]-t[:,0];p=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,p)
  ok=np.abs(det)>1e-12;inv=1./np.where(ok,det,1.);sv=o-t[:,0]
  u=np.einsum('ij,ij->i',sv,p)*inv;q=np.cross(sv,e1);v=np.einsum('ij,ij->i',d,q)*inv;dist=np.einsum('ij,ij->i',e2,q)*inv
  hit=ok&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(dist>0)&(dist<max_mm)
  np.minimum.at(out,r[hit],dist[hit])
 return out

def measure(mesh):
 """Sample points with raw and smoothed thickness (mm). Raw thickness is capped at RAY_MAX_MM (a ray that leaves the part
 reads RAY_MAX_MM). The cap is at least 2 x MIN_WALL_MM, so it cannot change whether a neighbourhood median is thin."""
 count=int(min(MAX_SAMPLES,max(200,mesh.area/SAMPLE_AREA_MM2)))
 pts,face=trimesh.sample.sample_surface_even(mesh,count,seed=SEED)
 inward=-mesh.face_normals[face]*(1. if mesh.volume>=0 else -1.)   # inward even for an inverted winding
 origins=pts+inward*RAY_OFFSET_MM
 raw=np.minimum(first_hit(mesh,origins,inward)+RAY_OFFSET_MM,RAY_MAX_MM)
 smooth=np.array([np.median(raw[nb])for nb in cKDTree(pts).query_ball_point(pts,NEIGHBOUR_MM)])
 return dict(points=pts,raw=raw,smoothed=smooth,areaPerSampleMm2=float(mesh.area/len(pts)))

def regions(meas):
 """Connected thin regions, largest first, of any size."""
 pts,sm=meas['points'],meas['smoothed'];thin=np.flatnonzero(sm<MIN_WALL_MM-MEASURE_TOL_MM)
 if not len(thin):return []
 n=len(thin);pairs=cKDTree(pts[thin]).query_pairs(NEIGHBOUR_MM,output_type='ndarray').reshape(-1,2)
 _,label=connected_components(coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(n,n)),directed=False);out=[]
 for k in np.unique(label):
  idx=thin[label==k];p=pts[idx];v=sm[idx]
  out.append(dict(bbox=[np.round(p.min(0),3).tolist(),np.round(p.max(0),3).tolist()],areaMm2=round(len(idx)*meas['areaPerSampleMm2'],2),
                  minMm=round(float(v.min()),3),medianMm=round(float(np.median(v)),3)))
 out.sort(key=lambda r:(-r['areaMm2'],r['bbox']));return out

def allowances(row):
 """Validated thinWallAllowances of a manifest row: [{bbox: [[x,y,z],[x,y,z]] print frame mm, minMm, reason}]."""
 a=row.get('thinWallAllowances',[]);key=row.get('key','part')
 assert isinstance(a,list),f'{key}: thinWallAllowances must be a list'
 out=[]
 for i,e in enumerate(a):
  b=np.array(e.get('bbox'),dtype=float);assert b.shape==(2,3) and np.all(b[0]<=b[1]),f'{key}: thinWallAllowances[{i}].bbox must be [[xmin,ymin,zmin],[xmax,ymax,zmax]] in the print frame'
  m=float(e.get('minMm',0));assert 0<m<MIN_WALL_MM,f'{key}: thinWallAllowances[{i}].minMm must be in (0, {MIN_WALL_MM})'
  assert str(e.get('reason','')).strip(),f'{key}: thinWallAllowances[{i}] needs a reason'
  out.append(dict(bbox=b.tolist(),minMm=m,reason=e['reason'].strip()))
 return out

def contains(box,reg):
 b=np.array(box);r=np.array(reg['bbox']);return bool(np.all(r[0]>=b[0]-BBOX_TOL_MM) and np.all(r[1]<=b[1]+BBOX_TOL_MM))

def check(mesh,allow=(),key='part'):
 """Wall check of a print-frame mesh. `regions` lists the regions of at least MIN_REGION_MM2 with their status (FAIL or
 allowed); smaller thin patches are only counted. `failures` holds one message per failing region."""
 meas=measure(mesh);found=regions(meas);used=set();failures=[];big=[];small=[]
 for reg in found:
  if reg['areaMm2']<MIN_REGION_MM2:small.append(reg['areaMm2']);continue
  hit=next((i for i,a in enumerate(allow)if contains(a['bbox'],reg) and reg['minMm']>=a['minMm']),None)
  if hit is None:
   reg['status']='FAIL'
   failures.append(f'{key}: wall thinner than {MIN_WALL_MM} mm: region bbox {reg["bbox"]} (print frame, mm), area {reg["areaMm2"]} mm2, '
                   f'median {reg["medianMm"]} mm, min {reg["minMm"]} mm. Thicken it, or add a thinWallAllowances entry with a reason')
  else:reg.update(status='allowed',allowance=hit);used.add(hit)
  big.append(reg)
 return dict(passed=not failures,minWallMm=MIN_WALL_MM,minRegionMm2=MIN_REGION_MM2,neighbourMm=NEIGHBOUR_MM,
             minSmoothedMm=round(float(meas['smoothed'].min()),3),samples=int(len(meas['points'])),areaPerSampleMm2=round(meas['areaPerSampleMm2'],4),
             regions=big,smallThinPatches=len(small),smallThinPatchMaxMm2=max(small,default=0.),
             allowancesUsed=[dict(allow[i],index=i)for i in sorted(used)],failures=failures)

if __name__=='__main__':
 for p in sys.argv[1:]:
  t=time.time();res=check(trimesh.load(p,force='mesh'),key=p);res.pop('failures');res['seconds']=round(time.time()-t,2)
  print(p);print(json.dumps(res,indent=1))
