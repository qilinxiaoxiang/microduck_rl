"""Read free-cloth surface after bounded normal-reset recorded motor replay.

This diagnostic does not establish a new controller success. No body or
vertex positions are assigned; it replays at most three seconds of motors.
"""
import argparse
import json
from pathlib import Path
from native_household import NativeHousehold

p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--seconds',type=float,default=1);a=p.parse_args()
if not .2<=a.seconds<=3:raise ValueError('Inspect only a bounded settled prefix')
e=json.loads(a.trace.read_text())
if e['config']['kind']!='sock':raise ValueError('Expected a cloth trace')
r=NativeHousehold(e['modelXml'],e['config'])
try:
    for row in e['rows']:
        r.step(row['ctrl'])
        if r.data.time>=a.seconds:break
    start=r.model.flex_vertadr[r.flex];count=r.model.flex_vertnum[r.flex];points=r.data.flexvert_xpos[start:start+count]
    sides=e['config']['sides'];rings=[]
    for ring in range(e['config']['rings']):
        group=points[ring*sides:(ring+1)*sides]
        rings.append(dict(ring=ring,vertices=group.tolist(),minimumZ=float(group[:,2].min()),maximumZ=float(group[:,2].max())))
    print(json.dumps(dict(scope='Recorded motor diagnostic, not new policy acceptance',seconds=r.data.time,summary=r.summary,rings=rings),indent=2))
finally:r.close()
