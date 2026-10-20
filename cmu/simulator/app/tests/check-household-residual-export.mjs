import fs from 'node:fs';
import {householdResidualActor} from '../src/game/household-residual-actor.js';
const request=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify(request.observations.map(obs=>householdResidualActor(request.actor,obs))));
