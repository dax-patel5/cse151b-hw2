# Running HW2 on UCSD Datahub

Pick the GPU-enabled notebook when spawning. Open a terminal.

## One-time setup

```bash
pip install --user tqdm pyyaml   # torch/torchvision/numpy/matplotlib are in the image
```

## Part 2 — the three seq-512 runs still outstanding (~1-2 h total on CUDA)

```bash
cd A2-Part2-Release
chmod +x run_experiments_datahub.sh
nohup ./run_experiments_datahub.sh rnn_seq512 lstm_seq512 noteacherforcing_seq512 \
      > seq512_queue.log 2>&1 &
tail -f seq512_queue.log        # Ctrl-C stops the tail, not the runs
```

Training is skipped for any experiment whose model file already exists in
`models/`, so only the seq-512 runs will train.

When done, download these back and drop them into the local
`A2-Part2-Release` folder (same relative paths):

- `models/trained_rnn_seq512_model.pth`
- `models/trained_lstm_seq512_model.pth`
- `models/trained_noteacherforcing_seq512_model.pth`
- `plots/rnn_seq512.png`, `plots/lstm_seq512.png`, `plots/noteacherforcing_seq512.png`
- `logs/rnn_seq512.log`, `logs/lstm_seq512.log`, `logs/noteacherforcing_seq512.log`

Note: seq-512 sequence tensors need ~10 GB RAM while loading — if the process
gets OOM-killed, respawn the Datahub container with more RAM.

## Part 1 — nothing needs training (models included); to re-verify:

```bash
cd A2-Part1-Release
python3 download.py     # fetches PASCAL VOC (~2 GB) into ./data
python3 train.py --experiment baseline
python3 train.py --experiment improved_baseline
python3 train.py --experiment experimental
```

Each just loads the shipped model and prints
`Final average pixel accuracy: ..., final average IoU: ...`.
