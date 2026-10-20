import * as ort from "onnxruntime-node";
import path from "node:path";
import {scene,mj,appDir} from "./physics-fixture.mjs";
import {POLICIES} from "../src/game/constants.js";
import {createWholeBodyCore} from "../src/game/whole-body-core.js";
export async function wholeBodyRuntime({mutate}={}) {
  const s=scene("legs",false,null,mutate),sessions={};
  const core=createWholeBodyCore(mj,s.model,s.data,{async infer(name,obs){
    sessions[name]??=await ort.InferenceSession.create(path.join(appDir,"public",POLICIES[name]),{executionProviders:["cpu"],intraOpNumThreads:1,interOpNumThreads:1});
    const out=await sessions[name].run({obs:new ort.Tensor("float32",obs,[1,61])});
    return out.actions.data;
  }});
  return {...core,xml:s.xml,async dispose(){for(const session of Object.values(sessions))await session.release();core.dispose();s.dispose();}};
}
