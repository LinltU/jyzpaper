#!/bin/bash
# Linux GPU detector grid: augmentation-first upstream anomaly detection.
PY="/home/test/miniconda3/envs/jyz_paper/bin/python"
WORK="/home/test/linliting/jyzmentor/code_reproduction/experiment_augmentation"
OUT="$WORK/runs/augmented_upstream_detection_gpu_20260910"
SYNTH="$WORK/runs/upstream_gpu_full_20260910"
LOG="$OUT/_grid_logs"
mkdir -p "$LOG"

# Prepare the split inside the detector output dir (same prepare() as generators).
"$PY" "$WORK/upstream_large_augmentation.py" --prepare --output "$OUT" > "$LOG/prepare.log" 2>&1

DETECTORS="${DETECTORS:-D3R IMDiffusion}"
AUGS="${AUGS:-real CSDI MTSCI IMDiffusion Diffusion-TS D3R}"
SEEDS="${SEEDS:-42 123 2024}"
DATES="${DATES:-250605 250626 250718}"
PARALLEL="${PARALLEL:-8}"

run_one() {
  det="$1"; aug="$2"; vol="$3"; seed="$4"; date="$5"
  tag="${det}_${aug}_${vol}_seed${seed}_${date}"
  logf="$LOG/$tag.log"
  if [ -f "$OUT/$det/$aug/$vol/seed$seed/$date/result.json" ]; then
    echo "skip $tag" >> "$LOG/done.log"
    return 0
  fi
  "$PY" "$WORK/augmented_upstream_detection.py" \
    --output "$OUT" --detector "$det" --aug "$aug" --volume "$vol" \
    --test-date "$date" --seed "$seed" --epochs 20 --steps 20 \
    --synth-root "$SYNTH" > "$logf" 2>&1
  code=$?
  if [ $code -eq 0 ]; then
    echo "OK $tag" >> "$LOG/done.log"
  else
    echo "FAIL $tag code=$code" >> "$LOG/done.log"
  fi
}
export -f run_one
export PY WORK OUT SYNTH LOG

for det in $DETECTORS; do
  for aug in $AUGS; do
    if [ "$aug" = "real" ]; then vols="1x"; else vols="10x large"; fi
    for vol in $vols; do
      for seed in $SEEDS; do
        for date in $DATES; do
          run_one "$det" "$aug" "$vol" "$seed" "$date" &
          while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do
            wait -n
          done
        done
      done
    done
  done
done
wait
echo "ALL_DETECTOR_GRID_DONE $(date)" >> "$LOG/done.log"
