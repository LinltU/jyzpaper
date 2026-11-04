#!/bin/bash
# Run a bounded batch of remaining detector runs in foreground with wait.
PY="/mnt/c/project/JYZmentor/code reproduction/runtime/python.exe"
WORK="C:/project/JYZmentor/code reproduction/experiment_augmentation"
OUT="$WORK/runs/augmented_upstream_detection_20260910"
OUT_LIN="/mnt/c/project/JYZmentor/code reproduction/experiment_augmentation/runs/augmented_upstream_detection_20260910"
BATCH="${1:-6}"
MAX="${2:-60}"
LOG="/mnt/c/project/JYZmentor/code reproduction/experiment_augmentation/runs/augmented_upstream_detection_20260910/_grid_logs"
mkdir -p "$LOG"
n=0
for det in D3R IMDiffusion; do
  for aug in real CSDI MTSCI IMDiffusion Diffusion-TS D3R; do
    if [ "$aug" = "real" ]; then vols="1x"; else vols="10x large"; fi
    for vol in $vols; do
      for s in 42 123; do
        for d in 250605 250626 250718; do
          tag="${det}_${aug}_${vol}_seed${s}_${d}"
          if [ -f "$OUT_LIN/$det/$aug/$vol/seed$s/$d/result.json" ]; then
            continue
          fi
          if [ $n -ge $MAX ]; then
            wait
            echo "BATCH_LIMIT_REACHED" >> "$LOG/done.log"
            exit 0
          fi
          "$PY" "$WORK/augmented_upstream_detection.py" \
            --output "$OUT" --detector "$det" --aug "$aug" --volume "$vol" \
            --test-date "$d" --seed "$s" --epochs 20 --steps 20 \
            > "$LOG/$tag.log" 2>&1 &
          n=$((n+1))
          if [ $((n % BATCH)) -eq 0 ]; then
            wait
          fi
        done
      done
    done
  done
done
wait
echo "ALL_REMAINING_DONE $(date)" >> "$LOG/done.log"
