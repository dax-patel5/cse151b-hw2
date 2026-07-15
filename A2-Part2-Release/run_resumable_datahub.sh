#!/bin/bash
# Datahub (Linux) variant of run_resumable.sh — uses system python3.
# Relaunches an experiment until main.py prints its final test loss, resuming
# from the per-epoch checkpoint in models/ckpt_<name>.pt after any interruption.
# Usage: ./run_resumable_datahub.sh <config_name> [<max_attempts>]
set -u
cd "$(dirname "$0")"
PY=python3
cfg=$1
max=${2:-15}
case "$cfg" in
    rnn*) exp=rnn ;;
    noteacherforcing*) exp=noteacherforcing ;;
    *) exp=lstm ;;
esac
mkdir -p logs
for i in $(seq 1 $max); do
    echo "=== $(date '+%H:%M:%S') attempt $i for $cfg ==="
    $PY main.py --experiment "$exp" --config "configs/$cfg.yaml" >> "logs/$cfg.log" 2>&1
    if grep -qa "Test set loss" "logs/$cfg.log"; then
        echo "=== $(date '+%H:%M:%S') completed $cfg: $(grep -ao 'Test set loss: [0-9.]*' logs/$cfg.log | tail -1) ==="
        exit 0
    fi
    echo "=== $(date '+%H:%M:%S') attempt $i died; resuming from checkpoint ==="
done
echo "=== gave up after $max attempts ==="
exit 1
