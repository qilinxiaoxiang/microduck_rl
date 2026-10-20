"""Native MuJoCo household physics with the browser's substep audit.

Only reset places bodies. During a trial the public mutation is a vector of
15 motor targets. Acceptance remains in the independent JavaScript verifier.
"""
from pathlib import Path
import mujoco
import numpy as np
from benchmark_native import ROOT, JOINTS, reset as reset_data
from native_contact_step import ContactStepper


def name_id(model, kind, name):
    return mujoco.mj_name2id(model, kind, name)


class NativeHousehold:
    def __init__(self, xml, config):
        if mujoco.mj_versionString()!='3.11.0':raise RuntimeError('Native/browser parity requires MuJoCo 3.11.0')
        self.xml, self.config = xml, config.copy()
        assets = {"assets/" + file.name: file.read_bytes()
                  for file in (ROOT / "app/public/robot/mjlab/meshes").glob("*.stl")}
        self.model = m = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = d = mujoco.MjData(m)
        self.qa = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINTS])
        self.va = np.array([m.jnt_dofadr[m.joint(n).id] for n in JOINTS])
        self.head = m.body("jaw_soft").id
        self.trunk = m.body("trunk_base").id
        self.jaw = m.jnt_qposadr[m.joint("beak_joint").id]
        self.flex = name_id(m, mujoco.mjtObj.mjOBJ_FLEX, "payload_flex")
        self.payload_joint = name_id(m, mujoco.mjtObj.mjOBJ_JOINT, "payload_freejoint")
        self.payload_geom = name_id(m, mujoco.mjtObj.mjOBJ_GEOM, "payload_geom")
        self.payload_body = m.body("payload").id
        self.p = int(m.jnt_qposadr[self.payload_joint]) if self.payload_joint >= 0 else None
        self.sites = {n: m.site(n).id for n in ["mouth_tip", "beak_upper_grip", "beak_lower_grip"]}
        self.stepper = ContactStepper(m, d, JOINTS)
        self.joint_ids = np.array([m.joint(n).id for n in JOINTS])
        self.default_pose = np.array([0,-.08726646259971647,-.457924,-.004940,.452984,.3490658503988659,.3490658503988659,0,0,0,.08726646259971647,.457924,.004940,-.452984],np.float32)
        self.gyro = m.sensor_adr[m.sensor('imu_ang_vel').id]
        self.feet = [m.geom(n).id for n in ['left_foot_collision','right_foot_collision']]
        self.beak_meshes = []
        for g in range(m.ngeom):
            if self.stepper.classes[g] & (4|8|16):
                mesh=m.geom_dataid[g];start=m.mesh_vertadr[mesh];count=m.mesh_vertnum[mesh]
                self.beak_meshes.append((g,m.mesh_vert[start:start+count].copy()))
        if self.flex >= 0:
            start, count = m.flex_vertadr[self.flex], m.flex_vertnum[self.flex]
            self.nodes = m.flex_vertbodyid[start:start+count]
            xs = m.body_pos[self.nodes, 0]
            self.ends = [np.flatnonzero(np.abs(xs-value)<1e-8) for value in [xs.min(), xs.max()]]
        def descendant(body):
            while body > 0 and body != self.payload_body:
                body = m.body_parentid[body]
            return body == self.payload_body
        free_cloth = self.p is None and all(m.body_jntnum[b] == 3 and all(
            m.jnt_type[m.body_jntadr[b]+k] == mujoco.mjtJoint.mjJNT_SLIDE
            and not m.jnt_limited[m.body_jntadr[b]+k] for k in range(3)) for b in self.nodes)
        self.model_audit = dict(freePayload=bool(free_cloth or self.p is not None and m.jnt_type[self.payload_joint] == mujoco.mjtJoint.mjJNT_FREE),
                               freeBase=bool(m.jnt_type[m.body_jntadr[self.trunk]] == mujoco.mjtJoint.mjJNT_FREE),
                               equalities=m.neq, mocapBodies=m.nmocap, payloadMass=float(m.body_subtreemass[self.payload_body]),
                               deformable=self.flex>=0, flexVertices=int(m.flex_vertnum[self.flex]) if self.flex>=0 else 0,
                               flexAllNodesFree=bool(self.flex>=0 and all(descendant(b) for b in self.nodes)),
                               pedestalPresent=name_id(m,mujoco.mjtObj.mjOBJ_GEOM,"payload_pedestal_geom")>=0,
                               terrainPresent=name_id(m,mujoco.mjtObj.mjOBJ_GEOM,"terrain")>=0,
                               floorHeight=float(m.geom_pos[m.geom("floor").id,2]),
                               jawTorqueLimit=m.actuator_forcerange[14].tolist(), timestep=m.opt.timestep)
        self.reset()

    def reset(self, config=None):
        candidate = {**self.config, **(config or {})}
        reset_data(self.model, self.data, candidate)
        self.config=candidate
        self.last_action=np.zeros(14,np.float32)
        self.summary = dict(maxPenetration=0., maxRobotPenetration=0., maxExternal=0.,
                            nonFootSupport=False, maxNonFootForce=0., nonFootImpulse=0., nonFootSeconds=0.,
                            steps=0, finite=True, solverReset=False, warningCounts={}, maxJointLimitExcess=0.)
        self.contact = dict(lowerForce=0.,upperForce=0.,feetForce=0.,floorForce=0.,nonMouthSupport=False,nonFootSupport=False)
        return self.snapshot()

    def step(self, targets):
        values=np.asarray(targets,dtype=np.float64)
        if values.shape!=(15,) or not np.all(np.isfinite(values)):
            raise ValueError("Expected 15 finite motor targets")
        self.data.ctrl[:]=values
        self.last_action=(values[:14]-self.default_pose.astype(np.float64)).astype(np.float32)
        v=self.stepper.step(round(.02/self.model.opt.timestep))
        self.contact=dict(samples=int(v[0]),lowerForce=v[1],upperForce=v[2],feetForce=v[3],floorForce=v[4],
                          nonFootForce=v[5],maxNonFootForce=v[6],maxPenetration=v[7],robotMaxPenetration=v[8],
                          nonMouthSupport=bool(v[9]),nonFootSupport=bool(v[10]),mouthRimForce=v[17])
        s=self.summary
        for key,index in [('maxPenetration',7),('maxRobotPenetration',8),('maxNonFootForce',6),('maxJointLimitExcess',11),('maxExternal',14)]:
            s[key]=max(s[key],float(v[index]))
        s['nonFootSupport'] |= bool(v[10]);s['finite'] &= bool(v[12]);s['solverReset'] |= bool(v[13])
        s['nonFootSeconds']+=float(v[15]);s['nonFootImpulse']+=float(v[16]);s['steps']+=int(v[19])
        if v[18]:s['warningCounts']['native']=int(v[18])
        return self.snapshot()

    def snapshot(self):
        m,d=self.model,self.data
        R=d.xmat[self.head].reshape(3,3)
        grasp=d.xpos[self.head]-R[:,0]*.016-R[:,2]*.075
        if self.p is not None:
            payload=d.qpos[self.p:self.p+7].tolist()
            g=self.payload_geom
            half=m.geom_size[g,0]+abs(d.geom_xmat[g,8])*m.geom_size[g,1]
            z=d.geom_xpos[g,2]
            shape=dict(minZ=float(z-half),maxZ=float(z+half))
        else:
            start,count=m.flex_vertadr[self.flex],m.flex_vertnum[self.flex]
            points=d.flexvert_xpos[start:start+count]
            ends=[points[idx,:2].mean(axis=0) for idx in self.ends]
            yaw=np.arctan2(*(ends[1]-ends[0])[::-1])
            payload=[*points[0].tolist(),float(np.cos(yaw/2)),0,0,float(np.sin(yaw/2))]
            shape=dict(minZ=float(points[:,2].min()-m.flex_radius[self.flex]),maxZ=float(points[:,2].max()+m.flex_radius[self.flex]))
        return dict(time=d.time,root=d.qpos[:7].tolist(),payload=payload,shape=shape,graspPoint=grasp.tolist(),
                    joint=d.qpos[self.qa].tolist(),ctrl=d.ctrl.tolist(),jaw=float(d.qpos[self.jaw]),
                    contact=self.contact.copy(),headQuat=d.xquat[self.head].tolist(),
                    **({'clothPoints':points.tolist()} if self.p is None else {}),
                    sites={name:d.site_xpos[index].tolist() for name,index in self.sites.items()})

    def close(self):
        self.stepper.close()

    def metadata(self):
        return dict(qa=self.qa.tolist(),va=self.va.tolist(),jntQposadr=self.model.jnt_qposadr.tolist(),jntRange=self.model.jnt_range.ravel().tolist())

    def controller_context(self, feedback=False):
        m,d=self.model,self.data
        w,x,y,z=d.xquat[self.trunk]
        gravity=[2*(w*y-x*z),-2*(w*x+y*z),2*(x*x+y*y)-1]
        obs=np.concatenate([d.sensordata[self.gyro:self.gyro+3],gravity,
                            d.qpos[self.qa]-self.default_pose,d.qvel[self.va],self.last_action]).astype(np.float32)
        context=dict(obsPrefix=obs.tolist(),bias=d.qfrc_bias.tolist(),axes=d.xaxis[self.joint_ids].tolist(),anchors=d.xanchor[self.joint_ids].tolist(),headOrientation=d.xmat[self.head].tolist())
        if not feedback:return context
        dofs=np.r_[np.arange(6),self.va];jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
        mujoco.mj_jacBody(m,d,jp,jr,self.head)
        kin=dict(headPoint=d.xpos[self.head].tolist(),headPosition=jp[:,dofs].tolist(),headRotation=jr[:,dofs].tolist(),feet=[])
        for g in self.feet:
            mujoco.mj_jacGeom(m,d,jp,jr,g)
            kin['feet'].append(dict(point=d.geom_xpos[g].tolist(),orientation=d.geom_xmat[g].tolist(),position=jp[:,dofs].tolist(),rotation=jr[:,dofs].tolist()))
        mujoco.mj_jacSubtreeCom(m,d,jp,self.trunk)
        kin['com']=dict(point=d.subtree_com[self.trunk].tolist(),position=jp[:,dofs].tolist())
        context['kinematics']=kin
        lowest=dict(height=float('inf'))
        for g,vertices in self.beak_meshes:
            world=vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            index=np.argmin(world[:,2]);height=world[index,2]
            if height<lowest['height']:lowest=dict(height=float(height),point=world[index].tolist(),geom=int(g))
        context['clearance']=lowest
        if self.payload_geom>=0 and self.config.get('surfaceSeek'):
            nearest=None;points=np.zeros(6)
            for geom in np.flatnonzero(self.stepper.classes&8):
                distance=mujoco.mj_geomDistance(m,d,int(geom),self.payload_geom,1.,points)
                if nearest is None or distance<nearest['distance']:
                    nearest=dict(distance=float(distance),point=points[:3].tolist(),objectPoint=points[3:].tolist(),geom=int(geom))
            context['surfaceGap']=nearest
        return context
