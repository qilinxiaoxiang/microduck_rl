// A tiny foam training block, intentionally high-contrast against the dark
// arena. The outer group owns the MJCF Z-up -> three.js Y-up conversion; the
// pose group inside takes raw free-joint position/quaternion values.
import * as THREE from "three";
import {
  PAYLOAD_CAP_HALF_X, PAYLOAD_CAP_HALF_Y, PAYLOAD_CAP_HALF_Z,
  PAYLOAD_SPAWN_DIST, PAYLOAD_LATERAL,
  PAYLOAD_PEDESTAL_HALF_XY, PAYLOAD_PEDESTAL_HALF_Z,
  SPAWN_X, SPAWN_Y,
} from "./constants.js";

export function createPayloadVisual() {
  const group = new THREE.Group();
  group.rotation.x = -Math.PI / 2;

  const pose = new THREE.Group();
  pose.visible = false;
  pose.userData.meshName = "foam_payload";

  const foamMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x26d9d0,
    roughness: 0.88,
    metalness: 0,
    clearcoat: 0.04,
  });
  const cap = new THREE.Mesh(
    new THREE.BoxGeometry(
      PAYLOAD_CAP_HALF_X * 2, PAYLOAD_CAP_HALF_Y * 2, PAYLOAD_CAP_HALF_Z * 2,
    ),
    foamMaterial,
  );
  pose.add(cap);

  // Two thin orange wraps make the small object legible at the default camera
  // distance without changing its collision size.
  const strapMat = new THREE.MeshStandardMaterial({
    color: 0xff6a21,
    roughness: 0.65,
    metalness: 0,
  });
  const band = PAYLOAD_CAP_HALF_Y * 0.42;
  const lift = PAYLOAD_CAP_HALF_Z + 0.00025;
  const strapA = new THREE.Mesh(
    new THREE.BoxGeometry(band, PAYLOAD_CAP_HALF_Y * 2.01, 0.0005),
    strapMat,
  );
  strapA.position.z = lift;
  const strapB = new THREE.Mesh(
    new THREE.BoxGeometry(PAYLOAD_CAP_HALF_X * 2.01, band, 0.0005),
    strapMat,
  );
  strapB.position.z = lift;
  pose.add(strapA, strapB);

  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(
      PAYLOAD_CAP_HALF_X * 2, PAYLOAD_CAP_HALF_Y * 2, PAYLOAD_CAP_HALF_Z * 2,
    )),
    new THREE.LineBasicMaterial({ color: 0x101018, transparent: true, opacity: 0.75 }),
  );
  pose.add(edges);

  const pedestal = new THREE.Mesh(
    new THREE.BoxGeometry(
      PAYLOAD_PEDESTAL_HALF_XY * 2,
      PAYLOAD_PEDESTAL_HALF_XY * 2,
      PAYLOAD_PEDESTAL_HALF_Z * 2,
    ),
    new THREE.MeshStandardMaterial({ color: 0x252832, roughness: 0.9, metalness: 0 }),
  );
  pedestal.position.set(
    SPAWN_X + PAYLOAD_SPAWN_DIST, SPAWN_Y + PAYLOAD_LATERAL, PAYLOAD_PEDESTAL_HALF_Z,
  );
  group.add(pedestal);

  group.add(pose);
  return { group, pose };
}
