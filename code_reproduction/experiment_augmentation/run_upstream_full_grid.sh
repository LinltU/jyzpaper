#!/bin/bash
PY="/mnt/c/project/JYZmentor/code reproduction/runtime/python.exe"
WORK="C:/project/JYZmentor/code reproduction/experiment_augmentation"
OUT="$WORK/runs/upstream_full_20260910"
LOG="/mnt/c/project/JYZmentor/code reproduction/experiment_augmentation/runs/upstream_full_20260910/_grid_logs"
mkdir -p "$LOG"
run_one() {
  method="$1"; seed="$2"; date="$3"
  tag="${method}_seed${seed}_${date}"
  logf="$LOG/$tag.log"
  if [ -f "$OUT/$method/seed$seed/$date/result.json" ]; then
    echo "skip $tag" >> "$LOG/done.log"
    return 0
  fi
  "$PY" "$WORK/upstream_large_augmentation.py" \
    --method "$method" --test-date "$date" --seed "$seed" \
    --epochs 80 --steps 20 --generate 20000 --sample-batch 256 \
    --output "$OUT" > "$logf" 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    echo "OK $tag" >> "$LOG/done.log"
  else
    echo "FAIL $tag code=$code" >> "$LOG/done.log"
  fi
}
export -f run_one
export PY WORK OUT LOG
printf '%s\n' CSDI MTSCI IMDiffusion Diffusion-TS D3R \
  | xargs -I{} -P5 bash -c 'for s in 42 123 2024; do for d in 250605 250626 250718; do run_one "{}" "$s" "$d"; done; done'
echo "ALL_GRID_DONE $(date)" >> "$LOG/done.log"
