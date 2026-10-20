"""Identical JavaScript controller and ONNX policies, native audited physics."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import hashlib
from native_household import NativeHousehold
from benchmark_native import ROOT


class ControllerBridge:
    def __init__(self):
        self.process=subprocess.Popen(['node','tests/native-controller-server.mjs'],cwd=ROOT/'app',stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def request(self,**request):
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('JavaScript controller process exited')
        result=json.loads(line)
        if 'error' in result:raise RuntimeError(result['error'])
        return result
    def close(self):
        if self.process.poll() is None:
            try:
                self.request(op='close')
                self.process.stdin.close()
                self.process.wait(timeout=10)
            except (BrokenPipeError, RuntimeError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill();self.process.wait()
        if not self.process.stdin.closed:self.process.stdin.close()
        self.process.stdout.close()


def source_hashes():
    files=[*(ROOT/'app/src/game').glob('*.js'),*(ROOT/'rl').glob('native_*'),ROOT/'rl/run_native_household.py',ROOT/'rl/benchmark_native.py',ROOT/'app/tests/native-controller-server.mjs',ROOT/'app/package-lock.json',
           ROOT/'app/public/policies/BEST_alpha_walking.onnx',ROOT/'app/public/policies/alpha_ground_pick.onnx',ROOT/'app/public/robot/mjlab/robot_allcollisions.xml',*(ROOT/'app/public/robot/mjlab/meshes').glob('*.stl')]
    return {str(file.relative_to(ROOT)):hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(files) if file.is_file()}


def run(config,output,quiet=False):
    if Path(output).exists():raise FileExistsError(output)
    bridge=ControllerBridge();runtime=None;hashes=source_hashes()
    try:
        initial=bridge.request(op='init',config=config)
        runtime=NativeHousehold(initial['xml'],initial['config']);rows=[];started=time.perf_counter();phase='manual'
        for k in range(config.get('ticks',1500)):
            result=bridge.request(op='step',snapshot=runtime.snapshot(),meta=runtime.metadata() if k==0 else None,
                                  context=runtime.controller_context(initial['config'].get('reachFeedback',False)),summary=runtime.summary)
            row=runtime.step(result['ctrl']);row['phase']=phase=result['phase'];row['alignment']=result.get('alignment');row['reachServo']=result.get('reachServo');rows.append(row)
            if phase in ['done','failed'] or not runtime.summary['finite']:break
        events=bridge.request(op='events')
        evidence=dict(config=initial['config'],phase=phase,rows=rows,summary=runtime.summary,modelAudit=runtime.model_audit,modelXml=runtime.xml,
                      wallSeconds=time.perf_counter()-started,provenance=dict(engine='MuJoCo 3.11.0 native',controller='shared JavaScript',inference='onnxruntime-node CPU',sources=hashes,sourcesUnchanged=hashes==source_hashes()),**events)
        evidence['acceptance']=bridge.request(op='evaluate',evidence={key:evidence[key] for key in ['phase','rows','summary','modelAudit']})
        if phase not in ['done','failed'] and not (config.get('holdUntilRelease') and phase=='carried'):
            evidence['incomplete']=True;evidence['failure']='Control-step budget exhausted before completion'
            evidence['acceptance']['accepted']=False;evidence['acceptance']['failures'].append(evidence['failure'])
        if not evidence['provenance']['sourcesUnchanged']:
            evidence['acceptance']['accepted']=False;evidence['acceptance']['failures'].append('Sources changed during trial')
        Path(output).write_text(json.dumps(evidence))
        if not quiet:print(json.dumps({k:v for k,v in evidence.items() if k not in ['rows','modelXml','provenance']}))
        return evidence
    finally:
        if runtime:runtime.close()
        bridge.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='{}');parser.add_argument('--output',required=True)
    args=parser.parse_args();run(json.loads(args.config),args.output)
