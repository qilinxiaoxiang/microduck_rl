"""Protocol tests with real MuJoCo state and a lightweight prediction stub."""
import json
from types import SimpleNamespace
import threading
import time
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import mujoco
import numpy as np
from household_planner_server import PlannerSession,PlannerHTTPServer,SPEC,model_text

class SessionTests(unittest.TestCase):
    def test_model_text_normalizes_only_declaration(self):
        xml='<mujoco><option timestep=".0000625"/></mujoco>'
        self.assertEqual(model_text('<?xml version="1.0"?>'+xml),xml)
        self.assertNotEqual(model_text(xml.replace('.0000625','.002')),xml)
    def setUp(self):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom size=".1"/></body></worldbody></mujoco>');d=mujoco.MjData(m)
        self.s=PlannerSession();self.s.runtime=SimpleNamespace(model=m,data=d,config={});self.s.token='fixture';self.s.touched=time.monotonic()
        self.s.planner=SimpleNamespace(last_time=-1,update=lambda stage:dict(stage=stage,ctrl=[0.]*14))
        values=np.zeros(mujoco.mj_stateSize(m,SPEC));mujoco.mj_getState(m,d,values,SPEC)
        self.request=dict(session='fixture',sequence=1,state=values.tolist())
    def test_state_and_order(self):
        result=self.s.plan(self.request);self.assertEqual(result['sequence'],1);self.assertEqual(self.s.runtime.data.qpos[2],1)
        with self.assertRaisesRegex(ValueError,'order'):self.s.plan(self.request)
    def test_bad_observations(self):
        for changes in [dict(session='wrong'),dict(state=[0]),dict(state=[float('nan')]*len(self.request['state']))]:
            with self.assertRaises(ValueError):self.s.plan({**self.request,**changes})
        self.assertEqual(self.s.sequence,0)
    def test_expired(self):
        self.s.touched-=601
        with self.assertRaisesRegex(ValueError,'expired'):self.s.plan(self.request)
    def test_stage_is_explicit_and_opt_in(self):
        for stage in ['teleport','transport']:
            with self.assertRaisesRegex(ValueError,'stage'):self.s.plan({**self.request,'stage':stage})
        self.assertEqual(self.s.sequence,0)
        self.s.runtime.config['dynamicsLift']=True
        self.assertEqual(self.s.plan({**self.request,'stage':'transport'})['plan']['stage'],'transport')

class HTTPTests(unittest.TestCase):
    def test_reject_cross_origin_and_non_json(self):
        with PlannerHTTPServer(('127.0.0.1',0)) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                for origin,kind,status in [('https://example.com','application/json',403),('http://127.0.0.1:5174','text/plain',415)]:
                    request=Request(f'http://127.0.0.1:{server.server_port}/init',data=b'{}',headers={'Origin':origin,'Content-Type':kind})
                    with self.assertRaises(HTTPError) as error:urlopen(request,timeout=5)
                    self.assertEqual(error.exception.code,status)
            finally:server.shutdown();thread.join()

if __name__=='__main__':unittest.main()
