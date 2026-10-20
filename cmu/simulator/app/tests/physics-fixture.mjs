import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";
import loadMujoco from "@mujoco/mujoco";
import { preparePhysicsXml } from "../src/game/physics-model.js";
import { makeGraspBenchXml } from "../src/game/grasp-bench-model.js";

export const appDir = fileURLToPath(new URL("../", import.meta.url));
export const meshDir = path.join(appDir, "public/robot/mjlab/meshes");
export const dom = new JSDOM("").window;
export const mj = await loadMujoco();
export function scene(variant = "legs", isolated = false, prop = null, mutate = null) {
  if (variant !== "legs") throw new Error(`unknown robot variant ${variant}: the duck has legs only`);
  let { xml, meshFiles } = preparePhysicsXml(fs.readFileSync(path.join(appDir, "public/robot/mjlab/robot_allcollisions.xml"), "utf8"), dom);
  if (isolated) xml = makeGraspBenchXml(xml, dom, prop ?? undefined);
  const vfs = new mj.MjVFS();
  if (mutate) xml = mutate(xml);
  for (const f of meshFiles) vfs.addBuffer(`assets/${f}`, fs.readFileSync(path.join(meshDir, f)));
  const model = mj.MjModel.from_xml_string(xml, vfs);
  const data = new mj.MjData(model);
  vfs.delete();
  return { model, data, xml, dispose() { data.delete(); model.delete(); } };
}
export const id = (model, type, name) => mj.mj_name2id(model, mj.mjtObj[`mjOBJ_${type}`].value, name);
export const name = (model, type, index) => mj.mj_id2name(model, mj.mjtObj[`mjOBJ_${type}`].value, index);
