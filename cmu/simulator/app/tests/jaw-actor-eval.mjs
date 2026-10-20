import fs from "node:fs";
import {jawActor} from "../src/game/jaw-rl.js";
const {actor,observations}=JSON.parse(fs.readFileSync(0,"utf8"));
console.log(JSON.stringify(observations.map(obs=>jawActor(actor,obs))));
