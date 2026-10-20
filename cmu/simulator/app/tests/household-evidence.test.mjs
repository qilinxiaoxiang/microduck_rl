import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {evaluateHouseholdTrial} from '../src/game/household-pick-acceptance.js';
const results=new URL('../test-results/',import.meta.url),summary=JSON.parse(fs.readFileSync(new URL(process.argv[2]??'household-validation.json',results)));
const tmp=fs.mkdtempSync(path.join(os.tmpdir(),'microduck-evidence-check-'));
execFileSync('unzip',['-q',new URL(summary.archive,results).pathname,'-d',tmp]);
for(const record of summary.records){
  const raw=fs.readFileSync(path.join(tmp,record.id)),d=JSON.parse(raw);
  assert.equal(createHash('sha256').update(raw).digest('hex'),record.sha256,record.id+' digest');
  if(d.incomplete||d.failure||record.invalidNegativeProtocol)continue;
  const result=evaluateHouseholdTrial(d,{requireRelease:!d.config.holdUntilRelease});
  assert.deepEqual(JSON.parse(JSON.stringify(result)),record.acceptance,record.id+' independent replay');
}
assert.ok(summary.records.some(r=>r.config.kind==='pen'&&r.acceptance.accepted));
assert.ok(summary.records.some(r=>r.config.kind==='sock'&&r.acceptance.accepted));
assert.ok(summary.records.some(r=>r.config.open&&!r.invalidNegativeProtocol&&!r.acceptance.accepted));
assert.ok(summary.records.some(r=>r.config.rootOffset&&!r.acceptance.accepted));
console.log(`PASS: ${summary.records.length} archived trajectory digests and independent readback, successes and failures retained`);
