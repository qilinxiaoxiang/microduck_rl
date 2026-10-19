"""CPU MuJoCo/ONNX rehearsal with voltage-controlled BAM XL330 actuators.

Changes are confined to generated models; no robot or cloud compute is contacted.
"""
from pathlib import Path
import argparse, collections, copy, hashlib, json, math, sys, time
import numpy as np
import mujoco
import onnxruntime as ort
from bam.model import load_model
from bam.mujoco import MujocoController

P = Path(__file__).resolve().parent
SIM = P.parents[2]
MODEL_DIR = SIM / 'microduck_rl/src/mjlab_microduck/robot/microduck'
POLICY_DIR = SIM / 'microduck-simulator/app/public/policies'
HOME = np.array([0,-.0872664626,-.457924,-.004940,.452984,
                 .3490658504,.3490658504,0,0,0,.0872664626,.457924,.004940,-.452984])

def add_signed_inertias(m, parts):
    """Combine replacement masses in body coordinates with the parallel-axis theorem."""
    parallel=lambda p:np.dot(p,p)*np.eye(3)-np.outer(p,p)
    for body in sorted({p['body'] for p in parts}):
        i=m.body(body).id;mass=float(m.body_mass[i]);com=m.body_ipos[i].copy()
        R=np.zeros(9);mujoco.mju_quat2Mat(R,m.body_iquat[i]);R=R.reshape(3,3)
        origin_i=R@np.diag(m.body_inertia[i])@R.T+mass*parallel(com)
        first=mass*com
        for p in [p for p in parts if p['body']==body]:
            dm=p['mass_kg'];position=np.array(p['position_m'])
            if 'inertia_about_com_kg_m2' in p:di=np.array(p['inertia_about_com_kg_m2'])
            else:
                size=np.array(p.get('dimensions_m',[.02,.02,.02]))
                di=np.diag(dm/12*(np.sum(size**2)-size**2))
                Q=np.zeros(9);mujoco.mju_quat2Mat(Q,np.array(p.get('rotation_wxyz',[1,0,0,0])))
                Q=Q.reshape(3,3);di=Q@di@Q.T
            mass+=dm;first+=dm*position;origin_i+=di+dm*parallel(position)
        assert mass>0
        com=first/mass;inertia=origin_i-mass*parallel(com)
        values,vectors=np.linalg.eigh(inertia)
        assert np.all(values>0) and values.max()<=values.sum()-values.max()+1e-10,(body,values)
        if np.linalg.det(vectors)<0:vectors[:,0]*=-1
        quat=np.zeros(4);mujoco.mju_mat2Quat(quat,vectors.ravel())
        m.body_mass[i]=mass;m.body_ipos[i]=com;m.body_inertia[i]=values;m.body_iquat[i]=quat

def load(config):
    spec = mujoco.MjSpec.from_file(str(MODEL_DIR/'scene.xml'))
    if config.get('scene'):
        spec = mujoco.MjSpec.from_file(str(Path(config['scene']).resolve()))
    if config.get('geometry_manifest'):
        manifest=json.loads(Path(config['geometry_manifest']).read_text())
        for g in list(spec.geoms):
            mesh=g.meshname if hasattr(g,'meshname') else ''
            if mesh in manifest.get('remove_meshes',[]):spec.delete(g)
        for i,item in enumerate(manifest['parts']):
            parent=spec.body(item['body'])
            asset=spec.add_mesh(name=f'new_cad_{i}',file=item['mesh'],scale=[.001]*3)
            parent.add_geom(name=f'new_cad_visual_{i}',type=mujoco.mjtGeom.mjGEOM_MESH,
                            meshname=asset.name,contype=0,conaffinity=0,mass=0,
                            rgba=item.get('rgba',[.8,.5,.12,1]),group=2)
            if item.get('collision_box'):
                box=item['collision_box']
                parent.add_geom(name=f'new_collision_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,
                                pos=box['position'],size=box['half_size'],mass=0,
                                contype=1,conaffinity=1,group=3)
    bam = load_model(motor_name='xl330', model='m6')
    bam.actuator.kp = config.get('kp', 200)
    bam.actuator.vin = config.get('voltage', 5.0)
    bam.actuator.max_current = config.get('current_limit', 1.47)
    limit = min(bam.actuator.vin * bam.kt.value/bam.R.value,
                (bam.actuator.max_current or 100)*bam.kt.value)
    names = []
    for act in spec.actuators:
        target = act.target.name if hasattr(act.target,'name') else str(act.target)
        if target.startswith('passive_'): continue
        names.append(act.name)
        act.set_to_motor(); act.ctrllimited=False; act.forcelimited=True
        act.forcerange=[-limit,limit]; act.gear=[1,0,0,0,0,0]
        joint=spec.joint(target)
        joint.damping=np.zeros((3,1));joint.frictionloss=0
        joint.solref_friction=[-5e4,-2e2]
        joint.solimp_friction=[.99,.9999,.001,.5,2]
    assert len(names)==14
    # Optional explicitly labelled hypothetical ballast, not a CAD mass claim.
    for i,a in enumerate(config.get('ballast',[])):
        parent=spec.body(a['body'])
        b=parent.add_body(name=f'ballast_{i}',pos=a['position'])
        b.add_geom(name=f'ballast_geom_{i}',type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=a.get('half_size',[.01,.01,.01]),mass=a['mass'],
                   contype=0,conaffinity=0,rgba=[.95,.35,.1,1])
    m=spec.compile();m.opt.timestep=config.get('timestep',.0025)
    m.vis.global_.offwidth=960;m.vis.global_.offheight=720
    if 'friction' in config:
        floor=m.geom('floor').id;m.geom_friction[floor,0]=config['friction']
        for i in range(m.ngeom):
            if m.body(m.geom_bodyid[i]).name in ('ankle_left','ankle_right'):
                m.geom_friction[i,0]=config['friction']
    mass_parts=copy.deepcopy(config.get('mass_parts',[]))
    if config.get('mass_manifest'):
        mass_parts+=json.loads(Path(config['mass_manifest']).read_text())['parts']
    if config.get('remove_old_battery_kg',0):
        audit=json.loads((P.parent/'electrical/existing-model-mass-audit.json').read_text())
        old=next(x for x in audit['models']['official_current']['parts'] if x['id']=='np_f970')
        removed=config['remove_old_battery_kg']
        assert 0 <= removed < old['assumption_dependent_removal_upper_bound_g']/1000
        mass_parts.append(dict(body='trunk_base',mass_kg=-removed,
             position_m=old['uniform_mesh_com_in_body_m'],
             inertia_about_com_kg_m2=(-removed*np.array(old['uniform_mesh_inertia_per_kg_m2'])).tolist()))
    if mass_parts:add_signed_inertias(m,mass_parts)
    d=mujoco.MjData(m)
    controller=MujocoController(bam,names,m,d,
        vin_drop_resistance=config.get('wire_resistance',0.0),
        vin_min=config.get('voltage_floor',4.0))
    # Runtime inertia and BAM armature changes invalidate compiled derived constants.
    mujoco.mj_setConst(m,d)
    return m,d,controller,names,bam

def run(config, out):
    started=time.perf_counter()
    m,d,c,names,bam=load(config)
    rng=np.random.default_rng(config.get('seed',0))
    ji=np.array([m.actuator(n).trnid[0] for n in names])
    qa=m.jnt_qposadr[ji]; va=m.jnt_dofadr[ji]
    trunk=m.body('trunk_base').id
    gyro=m.sensor('imu_ang_vel').adr[0]
    d.qpos[:7]=[0,0,.12,1,0,0,0]
    d.qpos[qa]=HOME+rng.normal(0,config.get('joint_noise',.0),14)
    mujoco.mj_forward(m,d)
    c.q_target=d.qpos[qa].copy()
    so=ort.SessionOptions();so.intra_op_num_threads=1;so.inter_op_num_threads=1
    sessions={k:ort.InferenceSession(str(POLICY_DIR/f),sess_options=so,providers=['CPUExecutionProvider'])
              for k,f in [('walk','BEST_alpha_walking.onnx'),('stand','BEST_alpha_stand.onnx')]}
    if config.get('policy')=='community':
        legacy=ort.InferenceSession(str(SIM/'Microduck-build-tutorial/microduck/src/agents/walk.onnx'),sess_options=so,providers=['CPUExecutionProvider'])
        sessions={'walk':legacy,'stand':legacy}
    expected=sessions['walk'].get_modelmeta().custom_metadata_map['joint_names'].split(',')
    assert names==expected, (names,expected)
    last=np.zeros(14);delay=config.get('action_delay_ticks',0)
    queue=collections.deque([HOME.copy() for _ in range(delay+1)],maxlen=delay+1)
    sensor_queue=collections.deque(maxlen=config.get('sensor_delay_ticks',0)+1)
    heading_queue=collections.deque(maxlen=sensor_queue.maxlen)
    rows=[];fell=False;fall_time=None;bad_contact=0;peak_current=0;max_torque=0
    duration=config.get('duration',12);steps=round(duration/.02)
    substeps=round(.02/m.opt.timestep)
    video=None;renderer=None
    if config.get('video'):
        import imageio.v2 as imageio
        renderer=mujoco.Renderer(m,height=720,width=960)
        video=imageio.get_writer(str(out.with_suffix('.mp4')),fps=25,codec='libx264',quality=7)
    for tick in range(steps):
        t=tick*.02
        R=d.xmat[trunk].reshape(3,3)
        measured_R=R
        if config.get('imu_noise_deg'):
            from scipy.spatial.transform import Rotation
            measured_R=R@Rotation.from_rotvec(rng.normal(0,math.radians(config['imu_noise_deg']),3)).as_matrix()
        g=measured_R.T@np.array([0,0,-1.])
        yaw_measurement=math.atan2(measured_R[1,0],measured_R[0,0])
        if not heading_queue:heading_queue.extend([yaw_measurement]*heading_queue.maxlen)
        heading_queue.append(yaw_measurement)
        cmd=np.zeros(13)
        if config.get('mode','walk')=='walk' and 2 <= t < duration-2:
            cmd[:3]=config.get('command',[.15,0,0])
            if config.get('heading_hold'):
                yaw=heading_queue[0]
                limit=config.get('heading_rate_limit',.6)
                cmd[2]=np.clip(-config.get('heading_gain',1.5)*yaw,-limit,limit)
        session=sessions['walk'] if np.linalg.norm(cmd[:3])>.001 else sessions['stand']
        measured_gyro=d.sensordata[gyro:gyro+3].copy()+rng.normal(0,config.get('gyro_noise_rad_s',0),3)
        sensors=np.r_[measured_gyro,g,d.qpos[qa]-HOME,d.qvel[va]]
        if not sensor_queue:
            sensor_queue.extend([sensors.copy() for _ in range(sensor_queue.maxlen)])
        sensor_queue.append(sensors.copy())
        sensed=sensor_queue[0].copy()
        if config.get('predict_delayed_state') and sensor_queue.maxlen>1:
            from scipy.spatial.transform import Rotation
            age=(sensor_queue.maxlen-1)*.02
            sensed[3:6]=Rotation.from_rotvec(-sensed[:3]*age).apply(sensed[3:6])
            sensed[6:20]+=sensed[20:34]*age
        command=cmd[:3] if config.get('policy')=='community' else cmd
        obs=np.r_[sensed,last,command].astype(np.float32)
        assert obs.shape==(session.get_inputs()[0].shape[1],)
        last=session.run(None,{session.get_inputs()[0].name:obs[None]})[0].reshape(-1)
        queue.append(HOME+last);c.q_target=queue[0].copy()
        if config.get('push') and tick==round(5/.02):
            d.qvel[:2]+=config['push']
        current_samples=[]
        for sub in range(substeps):
            c.update();mujoco.mj_step(m,d)
            current_samples.append(float(np.sum(np.abs(d.ctrl))/bam.kt.value))
            max_torque=max(max_torque,float(np.max(np.abs(d.actuator_force))))
        peak_current=max(peak_current,max(current_samples))
        tilt=math.degrees(math.acos(float(np.clip(d.xmat[trunk].reshape(3,3)[2,2],-1,1))))
        if not np.all(np.isfinite(d.qpos)) or tilt>60 or d.xpos[trunk,2]<.04:
            fell=True
            if fall_time is None:fall_time=round(t+.02,3)
        contact_pairs=[]
        for contact in d.contact:
            ids=(int(contact.geom1),int(contact.geom2))
            n=[m.geom(i).name for i in ids]
            if 'floor' in n:
                other=ids[1] if n[0]=='floor' else ids[0]
                body=m.body(m.geom_bodyid[other]).name
                if body not in ('ankle_left','ankle_right'):bad_contact+=1
                contact_pairs.append(body)
        rows.append({'time':round(t+.02,4),'qpos':d.qpos.tolist(),'x':float(d.xpos[trunk,0]),'y':float(d.xpos[trunk,1]),
                     'height':float(d.xpos[trunk,2]),'tilt_deg':tilt,'command':cmd[:3].tolist(),
                     'motor_current_sum_proxy_a':float(np.mean(current_samples)),
                     'torques_nm':d.actuator_force.tolist(),'ground_bodies':sorted(set(contact_pairs)),
                     'com':d.subtree_com[trunk].tolist(),
                     'yaw_rad':math.atan2(d.xmat[trunk].reshape(3,3)[1,0],d.xmat[trunk].reshape(3,3)[0,0]),
                     'target_angles':c.q_target.tolist()})
        if video and tick%2==0:
            cam=mujoco.MjvCamera();cam.lookat=d.xpos[trunk].copy()+[0,0,.06]
            cam.distance=.65;cam.azimuth=135;cam.elevation=-12
            renderer.update_scene(d,camera=cam);video.append_data(renderer.render())
        if fell and not config.get('continue_after_fall',False):break
    if video:video.close();renderer.close()
    moving=[r for r in rows if 3<=r['time']<=duration-2]
    vx=(moving[-1]['x']-moving[0]['x'])/(moving[-1]['time']-moving[0]['time']) if len(moving)>1 else None
    local_v=None;heading=None
    if len(moving)>1:
        xy=np.array([[r['x'],r['y']] for r in moving]);yaw=np.array([r['yaw_rad'] for r in moving])
        local_v=float(np.sum(np.diff(xy,axis=0)*np.c_[np.cos(yaw[:-1]),np.sin(yaw[:-1])])/(moving[-1]['time']-moving[0]['time']))
        heading=float(np.max(np.abs(np.degrees(np.unwrap(yaw)))))
    warnings=[{'index':i,'number':int(w.number),'lastinfo':int(w.lastinfo)} for i,w in enumerate(d.warning) if w.number]
    result={'config':config,'validation_revision':'mass-constants-recomputed-v2','baseline_model':'official compiled inertias unless explicitly modified',
        'mass_kg':float(sum(m.body_mass)),'duration_completed_s':rows[-1]['time'],
        'fell':fell,'fall_time_s':fall_time,'max_tilt_deg':max(r['tilt_deg'] for r in rows),
        'mean_forward_velocity_m_s':vx,'peak_summed_motor_current_proxy_a':peak_current,
        'mean_body_forward_velocity_m_s':local_v,'max_abs_heading_deviation_deg':heading,
        'max_actuator_torque_nm':max_torque,'nonfoot_ground_contact_samples':bad_contact,
        'warnings':warnings,'wall_seconds':time.perf_counter()-started,
        'engine':mujoco.mj_versionString(),'rows':rows}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('config',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();run(json.loads(args.config.read_text()),args.output)
