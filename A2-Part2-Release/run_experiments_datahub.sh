#!/bin/bash
# Datahub (Linux) variant of run_experiments.sh — uses system python3.
# Usage: ./run_experiments_datahub.sh <config_name> [<config_name> ...]
# Each config_name refers to configs/<config_name>.yaml; logs go to logs/<config_name>.log
set -u
cd "$(dirname "$0")"
PY=python3
mkdir -p logs
for cfg in "$@"; do
    case "$cfg" in
        rnn*) exp=rnn ;;
        noteacherforcing*) exp=noteacherforcing ;;
        *) exp=lstm ;;
    esac
    echo "=== $(date '+%H:%M:%S') starting $cfg (experiment=$exp) ==="
    $PY main.py --experiment "$exp" --config "configs/$cfg.yaml" > "logs/$cfg.log" 2>&1
    echo "=== $(date '+%H:%M:%S') finished $cfg: $(grep 'Test set loss' logs/$cfg.log) ==="
done
