#!/bin/zsh
# [AI-assisted: Claude Code]
# Sequential experiment runner for A2 Part 2.
# Usage: ./run_experiments.sh <config_name> [<config_name> ...]
# Each config_name refers to configs/<config_name>.yaml; logs go to logs/<config_name>.log
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
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
