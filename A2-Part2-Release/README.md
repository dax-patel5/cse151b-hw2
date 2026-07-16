# PA3 - Character level LSTM/RNN for Shakespearean Text Generation (A2 Part 2)

## Setup

Python 3 with PyTorch, numpy, matplotlib, PyYAML, and tqdm. The
TinyShakespeare data is in `data/tiny_shakespeare.txt`.

## Files

- `shakespeare_dataset.py` — `ShakespeareDataset`, wraps the pre-built
  (sequence, next character) tensors.
- `shakespeare_rnn.py` — `RNNModel`: embedding → multi-layer `nn.RNN` →
  linear head on the final hidden state (predicts the next character).
- `shakespeare_lstm.py` — `LSTMModel` (same structure with `nn.LSTM`) and
  `LSTMModelNoTeacherForcing`, which has the identical architecture but a
  forward pass that feeds only the first ground-truth character and then feeds
  back the argmax of its own output at every subsequent time step.
- `train.py` — `train` (Adam, cross-entropy, gradient clipping, early stopping
  on validation loss, saves the best checkpoint to
  `models/trained_<name>_model.pth`, writes loss curves to `plots/`) and `eval`.
  For the no-teacher-forcing model the training loss is computed at every time
  step against the ground-truth sequence; for the teacher-forced models the
  loss is on the next character following the sequence.
- `main.py` — data pipeline (provided) + experiment dispatch.
- `generate.py` / `generate.ipynb` / `generate_samples.py` — text generation
  with a temperature parameter; writes `temp0.5.txt`, `temp1.txt`, `temp2.txt`.
- `configs/` — one yaml per experiment (model type, seq_len, hidden size, etc.).
  `base_rnn_config.yaml` / `base_lstm_config.yaml` describe the saved best
  models used by the autograder flags.

## Running (autograder flags)

The best models are saved in `models/` and are loaded (not retrained) by:

```bash
python main.py --experiment rnn                # loads models/trained_rnn_model.pth
python main.py --experiment lstm               # loads models/trained_lstm_model.pth
python main.py --experiment noteacherforcing   # loads models/trained_noteacherforcing_model.pth
```

Each prints `Test set loss: xxxxx` at the end.

## Reproducing the experiments

Every experiment in question 5 has its own config; run them via:

```bash
python main.py --experiment lstm --config configs/lstm_seq128.yaml
./run_experiments.sh rnn_seq16 rnn_seq128 rnn_seq512 \
                     lstm_seq16 lstm_seq128 lstm_seq512 lstm_seq128_hidden300 \
                     noteacherforcing_seq16 noteacherforcing_seq128 noteacherforcing_seq512
```

Loss curves are written to `plots/<name>.png` and logs to `logs/<name>.log`.

## Text generation (question 5d)

```bash
python generate_samples.py     # or run generate.ipynb
```

generates ~1000-character samples from the best performing LSTM at
temperatures 0.5, 1, and 2 into `temp0.5.txt`, `temp1.txt`, `temp2.txt`.

## AI usage disclosure

Code sections marked with `[AI-assisted: Claude Code]` comments were written
with the help of Claude Code (Fable 5), as permitted by the course policy.

Here are some of the prompts I used:

"do the programming portion of these assignments entirely. You have full
liberty to train and run the models to meet the specs required by the
homework. However, you must perfectly abide by the submission, code, accuracy
etc requirements that the assignment pdfs detail."

There were smaller prompts about running the seq-512 experiments on UCSD
Datahub (whose stock PyTorch had to be upgraded for the assigned GPU),
pausing/resuming training, and creating the submission zip. Two notable
debugging episodes are reflected in the code comments: a PyTorch/MPS crash on
a trailing batch of size 1 (fixed with `drop_last=True`) and per-epoch
checkpoint-resume added after long runs were killed by memory pressure.
I reviewed the plan before training started and the results after each
experiment. I had no code-quality related concerns and did not manually edit
code.
