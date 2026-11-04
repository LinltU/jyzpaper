#!/bin/bash
# Linux GPU grid runner for the fixed-version upstream generators.
PY="/home/test/miniconda3/envs/jyz_paper/bin/python"
WORK="/home/test/linliting/jyzmentor/code_reproduction/experiment_augmentation"
OUT="$WORK/runs/upstream_gpu_full_20260910"
LOG="$OUT/_grid_logs"
mkdir -p "$LOG"

METHODS="${METHODS:-CSDI MTSCI IMDiffusion Diffusion-TS D3R}"
SEEDS="${SEEDS:-42 123 2024}"
DATES="${DATES:-250605 250626 250718}"
PARALLEL="${PARALLEL:-6}"
EPOCHS="${EPOCHS:-80}"
STEPS="${STEPS:-20}"
GEN="${GEN:-20000}"

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
    --epochs "$EPOCHS" --steps "$STEPS" --generate "$GEN" --sample-batch 256 \
    --output "$OUT" > "$logf" 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    echo "OK $tag" >> "$LOG/done.log"
  else
    echo "FAIL $tag code=$code" >> "$LOG/done.log"
  fi
}
export -f run_one
export PY WORK OUT LOG EPOCHS STEPS GEN

for m in $METHODS; do
  for s in $SEEDS; do
    for d in $DATES; do
      run_one "$m" "$s" "$d" &
      while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do
        wait -n
      done
    done
  done
done
wait
echo "ALL_GRID_DONE $(date)" >> "$LOG/done.log"
