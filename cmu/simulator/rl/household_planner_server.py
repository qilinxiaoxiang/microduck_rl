"""Local-only browser prediction service; the browser remains the real world.

Each request loads a read-only browser observation into a separate native
replica. Predictions return motor targets, never replacement body states.
No credentials, cloud service, or shared training volume are involved.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from pathlib import Path
import secrets
import shutil
import tempfile
import threading
import time
import mujoco
import numpy as np
from native_household import NativeHousehold
from predictive_reach import PredictiveReach
from run_native_household import ControllerBridge,source_hashes,ROOT

ORIGINS={'http://127.0.0.1:5174','http://localhost:5174','http://127.0.0.1:5173','http://localhost:5173'}
SPEC=mujoco.mjtState.mjSTATE_INTEGRATION
def model_text(xml):
    # Browser XMLSerializer preserves the source XML declaration; JSDOM's
    # serializer omits it. Normalize that declaration only, not model fields.
    return re.sub(r'^\s*<\?xml[^?]*\?>\s*','',xml)

class PlannerSession:
    def __init__(self):
        self.runtime=None;self.planner=None;self.token=None;self.sequence=0;self.touched=0.
    def close(self):
        if self.planner:self.planner.close()
        if self.runtime:self.runtime.close()
        self.runtime=None;self.planner=None;self.token=None
    def init(self,request):
        config=request.get('config')
        if not isinstance(config,dict) or config.get('kind') not in ['pen','sock']:raise ValueError('Expected an explicit household configuration')
        if config.get('dt')!=.0000625:raise ValueError('This prototype requires its validated physics timestep')
        bridge=ControllerBridge()
        try:initial=bridge.request(op='init',config=config)
        finally:bridge.close()
        native_xml=model_text(initial['xml']);digest=hashlib.sha256(native_xml.encode()).hexdigest()
        if request.get('modelSha256')!=digest:
            browser_xml=model_text(request.get('modelXml',''))
            first=next((i for i,(a,b) in enumerate(zip(browser_xml,native_xml)) if a!=b),min(len(browser_xml),len(native_xml)))
            raise ValueError(f'Browser/native XML differs at {first}: browser={browser_xml[max(0,first-25):first+75]!r}, native={native_xml[max(0,first-25):first+75]!r}')
        candidate=NativeHousehold(initial['xml'],initial['config'])
        signature=dict(nq=candidate.model.nq,nv=candidate.model.nv,nu=candidate.model.nu,nbody=candidate.model.nbody,ngeom=candidate.model.ngeom,stateSize=mujoco.mj_stateSize(candidate.model,SPEC))
        if request.get('signature')!=signature:
            candidate.close();raise ValueError('Browser/native model layout differs')
        self.close();self.runtime=candidate;self.planner=PredictiveReach(candidate)
        self.token=secrets.token_urlsafe(24);self.sequence=0;self.touched=time.monotonic()
        self.journal=Path(tempfile.mkdtemp(prefix='microduck-browser-planner-'));hashes=source_hashes()
        for name in ['rl/household_planner_server.py','rl/predictive_reach.py','rl/prediction_stepper.py','rl/grasp_forecast.py','app/src/household-pick.js','app/src/household-predictive-client.js','app/src/household-drive.js']:
            hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in hashes:
            destination=self.journal/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,destination)
        (self.journal/'init.json').write_text(json.dumps(dict(config=initial['config'],modelXml=initial['xml'],modelSha256=digest,signature=signature,sources=hashes)))
        print(f'Prediction observation journal: {self.journal}',flush=True)
        return dict(session=self.token,signature=signature,modelSha256=digest,engine=mujoco.mj_versionString(),scope='Native replica predictions only; actual browser physics is never replaced')
    def plan(self,request):
        if not self.runtime or request.get('session')!=self.token:raise ValueError('Unknown prediction session')
        if time.monotonic()-self.touched>600:raise ValueError('Prediction session expired; reset the playground')
        if request.get('sequence')!=self.sequence+1:raise ValueError('Out-of-order prediction request')
        stage=request.get('stage','reach')
        if stage not in ['reach','transport'] or stage=='transport' and not self.runtime.config.get('dynamicsLift'):raise ValueError('Unexpected predictive stage')
        values=np.asarray(request.get('state'),dtype=np.float64)
        if values.shape!=(mujoco.mj_stateSize(self.runtime.model,SPEC),) or not np.isfinite(values).all():raise ValueError('Expected a finite complete integration-state observation')
        r=self.runtime
        # This is exclusively a prediction replica. The HTTP response has no
        # state-setting command and the browser does not load this data back.
        mujoco.mj_setState(r.model,r.data,values,SPEC)
        if np.any(r.data.qfrc_applied) or np.any(r.data.xfrc_applied):raise ValueError('External-force observations are outside this prototype')
        mujoco.mj_forward(r.model,r.data)
        if r.data.time<0 or self.sequence and r.data.time<self.planner.last_time:raise ValueError('Simulation clock moved backwards; create a new session')
        input_snapshot=r.snapshot() if hasattr(r,'snapshot') else None
        result=self.planner.update(stage);self.sequence+=1;self.touched=time.monotonic()
        if hasattr(self,'journal'):
            # Deliberately omit the ephemeral session token. Observations and
            # results suffice for replay; no account files are ever included.
            (self.journal/f'plan-{self.sequence:03d}.json').write_text(json.dumps(dict(sequence=self.sequence,state=values.tolist(),snapshot=input_snapshot,plan=result)))
        return dict(sequence=self.sequence,time=float(r.data.time),plan=result)

class PlannerHTTPServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address):
        super().__init__(address,Handler);self.session=PlannerSession();self.lock=threading.Lock()
    def server_close(self):
        super().server_close()
        with self.lock:self.session.close()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,status,data):
        raw=json.dumps(data).encode();self.send_response(status)
        if self.headers.get('Origin') in ORIGINS:self.send_header('Access-Control-Allow-Origin',self.headers['Origin'])
        self.send_header('Vary','Origin');self.send_header('Cache-Control','no-store');self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers()
        try:self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_OPTIONS(self):
        if self.headers.get('Origin') not in ORIGINS:return self.reply(403,dict(error='Origin not allowed'))
        self.send_response(204);self.send_header('Access-Control-Allow-Origin',self.headers['Origin']);self.send_header('Access-Control-Allow-Methods','POST');self.send_header('Access-Control-Allow-Headers','Content-Type');self.send_header('Access-Control-Allow-Private-Network','true');self.end_headers()
    def do_POST(self):
        if self.headers.get('Origin') not in ORIGINS:return self.reply(403,dict(error='Origin not allowed'))
        if self.headers.get('Content-Type')!='application/json':return self.reply(415,dict(error='JSON required'))
        try:size=int(self.headers.get('Content-Length','0'))
        except ValueError:return self.reply(400,dict(error='Invalid request length'))
        if not 0<size<=2_000_000:return self.reply(413,dict(error='Request length outside limits'))
        if not self.server.lock.acquire(blocking=False):return self.reply(409,dict(error='Another prediction is still running'))
        try:
            self.connection.settimeout(10);request=json.loads(self.rfile.read(size))
            if not isinstance(request,dict):raise ValueError('Expected an object')
            if self.path=='/init':result=self.server.session.init(request)
            elif self.path=='/plan':result=self.server.session.plan(request)
            else:return self.reply(404,dict(error='Unknown operation'))
            self.reply(200,result)
        except (ValueError,TypeError,KeyError) as error:self.reply(400,dict(error=str(error)))
        except Exception as error:self.reply(500,dict(error=f'Prediction failed: {type(error).__name__}: {error}'))
        finally:self.server.lock.release()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18766);a=p.parse_args()
    if not 1024<=a.port<=65535:raise ValueError('Use a local nonprivileged port')
    with PlannerHTTPServer(('127.0.0.1',a.port)) as server:
        print(f'Local prediction service listening on 127.0.0.1:{a.port}; CPU only, no cloud spending',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
