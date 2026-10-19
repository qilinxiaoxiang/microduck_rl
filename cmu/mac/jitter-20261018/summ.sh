#!/bin/zsh
# summ.sh DIR NAME... : one line per case from jitter_sim JSON + compare.py windows (rel. to walk start)
D=$1; shift
for f in "$@"; do
  echo "$f $(python3 -c "import json;d=json.load(open('$D/$f.json'));print({k:d[k] for k in ('fell_at','max_tilt','max_action','action_jumps','dist_x')})")"
  ~/.venvs/microduck-dyn/bin/python "${0:A:h}/compare.py" $D/$f.jsonl:first --win 0 3 --win 10 22 --win 22.3 26 --win 26 34 2>&1 | python3 -c "
import sys,json
for l in sys.stdin:
  d=json.loads(l)
  if 'win' not in d: print('   guard',d['guard_trip_s'],'end',d['end_s']); continue
  print('   ',d['win'],'leg_dq',d.get('leg_dq'),'head_dq',d.get('head_dq'),'amax',d.get('leg_amax'),d.get('head_amax'),'q/a',d.get('leg_q_over_a'),'fa',d.get('leg_act_f_median'),'gyro',d.get('gyro_rms'),d.get('gyro_f'),'per_s',d.get('leg_dq_per_s'),'g',d.get('growth_per_s'))"
done
