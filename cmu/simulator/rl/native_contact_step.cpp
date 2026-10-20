// P0: the same substep-level contact audit as whole-body-core.js, in native
// MuJoCo. This function steps physics and reads evidence; it never adds forces,
// changes poses, attaches objects, or changes model parameters.
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>

extern "C" int microduck_step(const mjModel* m, mjData* d, int steps,
    int payload_geom, int payload_flex, int head, const int* classes,
    const int* joints, int joint_count, double* out) {
  if (!m || !d || !out || !classes || !joints || steps < 1 || steps > 1000000) return -1;
  std::fill(out, out+20, 0.0); out[12] = 1;
  for (int i=0; i<m->nv; ++i) out[14] = std::max(out[14], std::abs(d->qfrc_applied[i]));
  for (int i=0; i<6*m->nbody; ++i) out[14] = std::max(out[14], std::abs(d->xfrc_applied[i]));
  for (int step=0; step<steps; ++step) {
    const double before=d->time;
    mj_step(m,d);
    if (d->time<before+m->opt.timestep*.5) {out[12]=0; out[13]=1; break;}
    double lower=0,upper=0,feet=0,floor=0,nonfoot=0,rim_force=0;
    for (int i=0; i<d->ncon; ++i) {
      const mjContact& c=d->contact[i];
      const bool first=(payload_geom>=0 && c.geom[0]==payload_geom)||(payload_flex>=0 && c.flex[0]==payload_flex);
      const bool second=(payload_geom>=0 && c.geom[1]==payload_geom)||(payload_flex>=0 && c.flex[1]==payload_flex);
      const int g0=c.geom[0],g1=c.geom[1];
      const int robot=g0>=0 && (classes[g0]&1)?g0:g1>=0 && (classes[g1]&1)?g1:-1;
      const int other_robot=robot==g0?g1:g0;
      const bool environment=robot>=0 && other_robot>=0 && m->geom_bodyid[other_robot]==0;
      if (!first && !second && !environment) continue;
      if (environment) out[8]=std::max(out[8], -c.dist);
      if (first || second) out[7]=std::max(out[7], -c.dist);
      if (c.dist>0 || c.efc_address<0) continue;
      mjtNum force[6]; mj_contactForce(m,d,i,force);
      if (environment) {
        if (classes[robot]&2) feet+=force[0];
        else if (force[0]>1e-5) nonfoot+=force[0];
      }
      if ((!first && !second) || (first && second) || force[0]<=1e-5) continue;
      const int other=first?g1:g0,category=other>=0?classes[other]:0;
      if (category&4) lower+=force[0];
      else if (category&8) upper+=force[0];
      else {
        bool rim=false;
        if (category&16) {
          const mjtNum* R=d->xmat+head*9;
          const mjtNum* p=d->xpos+head*3;
          const double x=c.pos[0]-p[0],y=c.pos[1]-p[1],z=c.pos[2]-p[2];
          const double local_y=x*R[1]+y*R[4]+z*R[7],local_z=x*R[2]+y*R[5]+z*R[8];
          rim=local_z<=-.0435 && std::abs(local_y)<=.0462;
          if (rim) rim_force+=force[0];
        }
        if (!rim) out[9]=1;
        if (other>=0 && m->geom_bodyid[other]==0) floor+=force[0];
      }
    }
    out[0]+=1;out[1]+=lower;out[2]+=upper;out[3]+=feet;out[4]+=floor;out[5]+=nonfoot;
    out[6]=std::max(out[6],nonfoot);out[17]+=rim_force;
    if (nonfoot>0) {out[10]=1;out[15]+=m->opt.timestep;}
    out[16]+=nonfoot*m->opt.timestep;
    for (int i=0;i<joint_count;++i) {
      const int j=joints[i];const double q=d->qpos[m->jnt_qposadr[j]];
      out[11]=std::max({out[11],m->jnt_range[2*j]-q,q-m->jnt_range[2*j+1]});
    }
    for (int i=0;i<m->nq;++i) if (!std::isfinite(d->qpos[i])) out[12]=0;
    out[19]+=1;
  }
  for (int i=0;i<mjNWARNING;++i) out[18]=std::max(out[18],double(d->warning[i].number));
  const double samples=std::max(1.0,out[0]);
  for (int i: {1,2,3,4,5,17}) out[i]/=samples;
  return 0;
}
