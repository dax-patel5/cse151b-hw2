from util import *
import gc
import torch
import torch.optim as optim
import torch.nn as nn
import os

from tqdm import tqdm

from shakespeare_lstm import LSTMModel, LSTMModelNoTeacherForcing
from shakespeare_rnn import RNNModel


def _batch_loss(model, criterion, x, y):
    """Cross-entropy loss for one batch, handling both training regimes.
    [AI-assisted: Claude Code]


    Teacher-forced models emit logits only for the character following the
    sequence, so the loss is against y. The no-teacher-forcing model emits a
    prediction at every time step, so the loss is against the ground truth at
    every position (x shifted left by one, then y).
    """
    if isinstance(model, LSTMModelNoTeacherForcing):
        logits = model(x, return_all=True)                      # (B, T, V)
        targets = torch.cat([x[:, 1:], y.unsqueeze(1)], dim=1)  # (B, T)
        return criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    logits = model(x)                                           # (B, V)
    return criterion(logits, y)


def train(model, device, train_dataloader, val_dataloader, config):
    # [AI-assisted: Claude Code]

    # for autograding purposes - your train should also save your best model to the ./models folder
    experiment = config.get('name', config.get('experiment', 'model'))
    models_loc = "./models"
    os.makedirs(models_loc, exist_ok=True)
    model_path = os.path.join(models_loc, f"trained_{experiment}_model.pth")

    epochs = config['epochs']
    patience = config.get('patience', 3)
    lr = config.get('lr', 0.002)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses, val_losses = [], []

    # Per-epoch checkpointing so an interrupted run (e.g. the process getting
    # OOM-killed mid-run on long-sequence experiments) resumes instead of
    # starting over. The checkpoint is deleted once training completes.
    ckpt_path = os.path.join(models_loc, f"ckpt_{experiment}.pt")
    start_epoch = 0
    pending_train_loss = None  # set when the crash happened between train and eval
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        train_losses, val_losses = ck['train_losses'], ck['val_losses']
        best_val_loss = ck['best_val_loss']
        epochs_without_improvement = ck['no_improve']
        start_epoch = ck['epoch']
        if ck['phase'] == 'trained':
            pending_train_loss = ck['train_loss']
        else:
            start_epoch += 1
        print(f"Resumed from checkpoint: epoch {start_epoch}, phase {ck['phase']}")

    def save_ckpt(phase, epoch, train_loss=None):
        torch.save({'phase': phase, 'epoch': epoch, 'train_loss': train_loss,
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'train_losses': train_losses, 'val_losses': val_losses,
                    'best_val_loss': best_val_loss, 'no_improve': epochs_without_improvement},
                   ckpt_path)

    for epoch in range(start_epoch, epochs):
        if pending_train_loss is not None:
            train_loss = pending_train_loss
            pending_train_loss = None
        else:
            model.train()
            running_loss, n_batches = 0.0, 0
            for x, y in tqdm(train_dataloader, desc=f"epoch {epoch + 1}/{epochs}", leave=False):
                x, y = x.to(device).long(), y.to(device)
                optimizer.zero_grad()
                loss = _batch_loss(model, criterion, x, y)
                loss.backward()
                # vanilla RNNs (and long self-fed rollouts) are prone to exploding
                # gradients, so clip before stepping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1

            train_loss = running_loss / n_batches
            save_ckpt('trained', epoch, train_loss)

        # release cached GPU blocks before eval reshapes allocations — long
        # rollouts (seq_len 512) otherwise spike memory at the epoch boundary
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_loss = eval(model, device, val_dataloader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                save_ckpt('evaled', epoch)
                break
        save_ckpt('evaled', epoch)

    plot_losses(train_losses, val_losses, config.get('name', experiment))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    # leave the best weights (already saved to disk) loaded on the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    return train_losses, val_losses


def eval(model, device, val_dataloader):
    # [AI-assisted: Claude Code]

    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for x, y in val_dataloader:
            x, y = x.to(device).long(), y.to(device)
            logits = model(x)               # (B, V): next-character logits
            total_loss += criterion(logits, y).item()
            n_batches += 1
    val_loss = total_loss / n_batches

    return val_loss
