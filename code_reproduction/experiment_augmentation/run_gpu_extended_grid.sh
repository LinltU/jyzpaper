#!/bin/bash
# Linux GPU grid for the mechanism-adapted remaining papers.
PY="/home/test/miniconda3/envs/jyz_paper/bin/python"
WORK="/home/test/linliting/jyzmentor/code_reproduction/experiment_augmentation"
OUT="$WORK/runs/all_papers_extended_20260910"
LOG="$OUT/_grid_logs"
mkdir -p "$LOG"

"$PY" "$WORK/upstream_large_augmentation.py" --prepare --output "$OUT" > "$LOG/prepare.log" 2>&1

GENS="TimeGrad NsDiff FIDE PaD-TS WaveStitch TabDDPM CCDM DiffPLF TSDE SSSD CoSTI SSD-TS MCD-TSF"
DETS="DiffAD ICDiffAD TIMEADDM ProDiffAD TSADC NGLS-Diff DiffTAD diffusionae C3DAD"
SEEDS="42 123 2024"
DATES="250605 250626 250718"
PARALLEL="${PARALLEL:-8}"

run_gen() {
  m="$1"; s="$2"; d="$3"
  tag="gen_${m}_seed${s}_${d}"
  if [ -f "$OUT/generators/$m/seed$s/$d/result.json" ]; then echo "skip $tag" >> "$LOG/done.log"; return 0; fi
  "$PY" "$WORK/all_papers_extended.py" --mode generator --method "$m" --test-date "$d" --seed "$s" \
    --epochs 60 --steps 20 --generate 20000 --output "$OUT" > "$LOG/$tag.log" 2>&1
  if [ $? -eq 0 ]; then echo "OK $tag" >> "$LOG/done.log"; else echo "FAIL $tag" >> "$LOG/done.log"; fi
}
run_det() {
  m="$1"; s="$2"; d="$3"
  tag="det_${m}_seed${s}_${d}"
  if [ -f "$OUT/detectors/$m/seed$s/$d/result.json" ]; then echo "skip $tag" >> "$LOG/done.log"; return 0; fi
  "$PY" "$WORK/all_papers_extended.py" --mode detector --method "$m" --test-date "$d" --seed "$s" \
    --epochs 60 --steps 20 --output "$OUT" > "$LOG/$tag.log" 2>&1
  if [ $? -eq 0 ]; then echo "OK $tag" >> "$LOG/done.log"; else echo "FAIL $tag" >> "$LOG/done.log"; fi
}
export -f run_gen run_det
export PY WORK OUT LOG

for m in $GENS; do for s in $SEEDS; do for d in $DATES; do
  run_gen "$m" "$s" "$d" &
  while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
done; done; done
wait

for m in $DETS; do for s in $SEEDS; do for d in $DATES; do
  run_det "$m" "$s" "$d" &
  while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
done; done; done
wait
echo "ALL_EXTENDED_DONE $(date)" >> "$LOG/done.log"
