// Independent post-run review; missing force evidence is never positive contact.
export function stableGroundLanding(rows){let count=0,previous=null;for(const s of rows.toReversed()){
 const resting=["release","done"].includes(s.phase)&&s.jaw>.30&&s.contact.upperForce<.002&&s.contact.lowerForce<.002&&s.contact.floorForce>.001&&Math.abs(s.shape.minZ)<.0005;
 if(!resting||previous!==null&&Math.abs(previous-s.time-.02)>1e-6)break;
 count++;previous=s.time;
}return count*.02;}
