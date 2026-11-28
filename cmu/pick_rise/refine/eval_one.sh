cd /home/linok/Downloads/microduck_rl-dock
OUT=cmu/pick_rise/refine; name=$1; onnx=$2
for c in SN1 RC SN0 SN2 SN3 RCSN2 RCSN3; do
  extra=(); case $c in SN1|RC) extra=(--grid) ;; esac
  uv run --no-sync python cmu/kneel/kr_eval.py --onnx "$onnx" --ball-only --num-envs 512 "${extra[@]}" \
    --task "Mjlab-KneelRise-DockFitC-Ball-$c-MicroDuck-CMU-Current" --cases w1,w5,w3_lat55 \
    --out "$OUT/results/$name/$c.json" > "$OUT/results/${name}_$c.log" 2>&1
  echo "$(date +%H:%M) $name $c done"
done
