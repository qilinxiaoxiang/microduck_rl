// Retain complete success/failure traces and the exact local implementation.
// Refuses to overwrite an earlier archive; never uploads or pushes anything.
import fs from "node:fs";
import path from "node:path";
import {execFileSync} from "node:child_process";
import {createHash} from "node:crypto";
const root=path.resolve(import.meta.dirname,".."),archive=process.argv[2]??"app/test-results/ground-ball-case-20260904.zip";
const validation=JSON.parse(fs.readFileSync(path.join(root,"rl/results/ground-validation.json")));
if(!validation.accepted)throw Error("Ground case review has not passed");
if(fs.existsSync(path.resolve(root,archive)))throw Error("Archive already exists; choose a new archive path");
const runEntries=fs.readdirSync(path.join(root,"rl/runs")).filter(n=>n.startsWith("ground-")).map(n=>`rl/runs/${n}`);
const inputs=[...runEntries,"app/src/game","app/src/table-pick.js","app/tests","app/ground-pick.html","app/package.json","app/package-lock.json","app/vite.config.js","app/public/policies/jaw-ppo.json","app/public/policies/jaw-ppo-initial.json","app/test-results/physics-validation.json","rl/GROUND_RESULTS.md","rl/accept_ground_case.mjs","rl/ground_review_helpers.mjs","rl/evaluate_ground.mjs","rl/pack_ground_case.mjs","rl/results/ground-validation.json","Play-Jaw-RL.command"];
fs.mkdirSync(path.dirname(path.resolve(root,archive)),{recursive:true});
execFileSync("zip",["-q","-r",archive,...inputs],{cwd:root});
execFileSync("unzip",["-tq",archive],{cwd:root});
const digest=p=>createHash("sha256").update(fs.readFileSync(path.resolve(root,p))).digest("hex");
const result={generatedAt:new Date().toISOString(),archive,archiveSha256:digest(archive),archiveBytes:fs.statSync(path.resolve(root,archive)).size,validationSha256:digest("rl/results/ground-validation.json"),actorSha256:digest("app/public/policies/jaw-ppo.json"),inputs,scope:"Local evidence only; includes rejected development/mesh cases. No retraining, publication or upload."};
fs.writeFileSync(path.join(root,"rl/results/ground-artifact-hashes.json"),JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result,null,2));
