#!/bin/zsh
# [AI-assisted: Claude Code]
# Repeatedly (re)launch one experiment until main.py prints its final test
# loss, relying on the per-epoch checkpoint in train.py to resume after the
# process gets OOM-killed. Usage: ./run_resumable.sh <config_name> <max_attempts>
set -u
cd "$(dirname "$0")"
PY=../.venv/bin/python
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
