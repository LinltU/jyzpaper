#!/bin/bash
PY="/mnt/c/project/JYZmentor/code reproduction/runtime/python.exe"
WORK="C:/project/JYZmentor/code reproduction/experiment_augmentation"
OUT="$WORK/runs/augmented_upstream_detection_20260910"
LOG="$OUT/_grid_logs"
mkdir -p "$LOG"
run_one() {
  det="$1"; aug="$2"; vol="$3"; seed="$4"; date="$5"
  tag="${det}_${aug}_${vol}_seed${seed}_${date}"
  if [ -f "$OUT/$det/$aug/seed$seed/$date/result.json" ]; then
    echo "skip $tag" >> "$LOG/done.log"
    return 0
  fi
  logf="$LOG/$tag.log"
  "$PY" "$WORK/augmented_upstream_detection.py" \
    --output "$OUT" --detector "$det" --aug "$aug" --volume "$vol" \
    --test-date "$date" --seed "$seed" --epochs 20 --steps 20 \
    > "$logf" 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    echo "OK $tag" >> "$LOG/done.log"
  else
    echo "FAIL $tag code=$code" >> "$LOG/done.log"
  fi
}
export -f run_one
export PY WORK OUT LOG
# 132 runs: 2 detectors x (real + 5 augs x 2 volumes) x 3 dates x 2 seeds
printf '%s\n' D3R IMDiffusion | xargs -I{} -P2 bash -c '
  for a in real CSDI MTSCI IMDiffusion Diffusion-TS D3R; do
    if [ "$a" = "real" ]; then vols="1x"; else vols="10x large"; fi
    for v in $vols; do
      for s in 42 123; do
        for d in 250605 250626 250718; do
          run_one "{}" "$a" "$v" "$s" "$d"
        done
      done
    done
  done'
echo "ALL_DETECTION_GRID_DONE $(date)" >> "$LOG/done.log"
