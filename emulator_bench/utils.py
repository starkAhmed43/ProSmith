"""
utils.py — ProSmith emulator_bench shared metrics, early stopping, and epoch runners.

Metrics: PCC, SCC, R2, RMSE, MSE, MAE  (same interface as CataPro emulator_bench)
EarlyStopping: same logic as CataPro (patience + min_delta on val loss)
run_a_training_epoch / run_an_eval_epoch: adapted for MM_TN forward signature.
"""

import os
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def mse(y_true, y_pred):
    return np.mean(np.square(y_pred - y_true))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))


def pcc(y_true, y_pred):
    fsp = y_pred - np.mean(y_pred)
    fst = y_true - np.mean(y_true)
    dev_p = np.std(y_pred)
    dev_t = np.std(y_true)
    if dev_p == 0 or dev_t == 0:
        return 0.0
    return float(np.mean(fsp * fst) / (dev_p * dev_t))


def scc(y_true, y_pred):
    return pcc(rankdata(y_true), rankdata(y_pred))


def r2_score(y_true, y_pred):
    numerator = np.sum(np.square(y_true - y_pred))
    denominator = np.sum(np.square(y_true - np.mean(y_true)))
    if denominator == 0:
        return 0.0
    return float(1 - numerator / denominator)


def evaluate(y_true, y_pred):
    return (
        pcc(y_true, y_pred),
        scc(y_true, y_pred),
        r2_score(y_true, y_pred),
        rmse(y_true, y_pred),
        mse(y_true, y_pred),
        mae(y_true, y_pred),
    )


# ---------------------------------------------------------------------------
# CSV result writers  (same format as CataPro emulator_bench)
# ---------------------------------------------------------------------------

def out_results(values, file_path):
    """values: np.array of shape (7,) = [pcc,scc,r2,rmse,mse,mae,loss]"""
    columns = ["valid_pcc", "valid_scc", "valid_r2", "valid_rmse", "valid_mse", "valid_mae", "valid_loss"]
    df = pd.DataFrame(values.reshape(1, -1), columns=columns)
    df.to_csv(file_path, float_format="%.5f", index=False)


def write_logfile(epoch, record_data, logfile):
    if epoch == 0 and os.path.exists(logfile):
        os.remove(logfile)
    values = np.array(record_data).reshape(epoch + 1, -1)
    columns = [
        "epoch",
        "train_pcc", "train_scc", "train_r2", "train_rmse", "train_mse", "train_mae", "train_loss",
        "valid_pcc", "valid_scc", "valid_r2", "valid_rmse", "valid_mse", "valid_mae", "valid_loss",
    ]
    df = pd.DataFrame(values, index=list(range(epoch + 1)), columns=columns)
    df.to_csv(logfile, float_format="%.4f")


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = None
        self.count_epoch = 0
        self.stop = False
        self.is_bestmodel = False

    def check(self, epoch, cur_loss):
        if epoch == 0:
            self.min_loss = cur_loss
            self.count_epoch = 1
            self.is_bestmodel = True
        else:
            if cur_loss < self.min_loss - self.min_delta:
                self.min_loss = cur_loss
                self.count_epoch = 0
                self.is_bestmodel = True
            else:
                self.count_epoch += 1
                self.is_bestmodel = False

        if self.count_epoch >= self.patience:
            self.stop = True

        return self.is_bestmodel, self.stop

    def state_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_loss": self.min_loss,
            "count_epoch": self.count_epoch,
            "stop": self.stop,
            "is_bestmodel": self.is_bestmodel,
        }

    def load_state_dict(self, state):
        self.patience = int(state.get("patience", self.patience))
        self.min_delta = float(state.get("min_delta", self.min_delta))
        self.min_loss = state.get("min_loss", self.min_loss)
        self.count_epoch = int(state.get("count_epoch", 0))
        self.stop = bool(state.get("stop", False))
        self.is_bestmodel = bool(state.get("is_bestmodel", False))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

mse_loss = nn.MSELoss()


# ---------------------------------------------------------------------------
# Epoch runners  (MM_TN forward signature)
# ---------------------------------------------------------------------------

def _move_batch(batch, device):
    return [b.to(device) for b in batch]


def _autocast_context(device, autocast_dtype=None):
    if autocast_dtype is not None and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)
    return nullcontext()


def run_a_training_epoch(
    model,
    data_loader,
    optimizer,
    device,
    scaler=None,
    autocast_dtype=None,
    rank=0,
    progress_desc=None,
    progress_disable=True,
    grad_accum_steps=1,
):
    """
    Trains for one epoch. Returns np.array([pcc,scc,r2,rmse,mse,mae,avg_loss]).
    Batch format from ProSmithDataset collate:
      smiles_emb  [B, max_smi, 600]
      smiles_attn [B, max_smi]
      protein_emb [B, max_prot, 1280]
      protein_attn[B, max_prot]
      labels      [B, 1]
    """
    model.train()
    total_loss = 0.0
    y_label, y_pred = [], []

    batch_iter = data_loader
    if progress_desc is not None:
        from tqdm.auto import tqdm

        batch_iter = tqdm(data_loader, desc=progress_desc, unit="batch", leave=False, disable=progress_disable)

    optimizer.zero_grad()
    num_batches = 0
    for batch_idx, batch in enumerate(batch_iter):
        smiles_emb, smiles_attn, protein_emb, protein_attn, labels = _move_batch(batch, device)
        with _autocast_context(device, autocast_dtype=autocast_dtype):
            outputs = model(
                smiles_emb=smiles_emb,
                smiles_attn=smiles_attn,
                protein_emb=protein_emb,
                protein_attn=protein_attn,
                device=device,
                gpu=rank,
            )  # [B, 1]

            loss = mse_loss(outputs, labels.float())
            loss_for_backward = loss / max(1, grad_accum_steps)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = ((batch_idx + 1) % max(1, grad_accum_steps) == 0) or ((batch_idx + 1) == len(data_loader))
        if should_step:
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if progress_desc is not None:
            batch_iter.set_postfix(loss=f"{loss.item():.4f}", accum=f"{grad_accum_steps}")

        total_loss += loss.item()
        y_label.append(labels.squeeze(-1).detach().float().cpu().numpy())
        y_pred.append(outputs.squeeze(-1).detach().float().cpu().numpy())
        num_batches += 1

    y_label = np.concatenate(y_label)
    y_pred = np.concatenate(y_pred)
    _pcc, _scc, _r2, _rmse, _mse, _mae = evaluate(y_label, y_pred)
    return np.array([_pcc, _scc, _r2, _rmse, _mse, _mae, total_loss / max(num_batches, 1)])


def run_an_eval_epoch(model, data_loader, device, rank=0, autocast_dtype=None):
    """
    Returns (y_pred, y_label, np.array([pcc,scc,r2,rmse,mse,mae,avg_loss])).
    """
    model.eval()
    total_loss = 0.0
    y_label, y_pred = [], []

    with torch.no_grad():
        for batch in data_loader:
            smiles_emb, smiles_attn, protein_emb, protein_attn, labels = _move_batch(batch, device)

            with _autocast_context(device, autocast_dtype=autocast_dtype):
                outputs = model(
                    smiles_emb=smiles_emb,
                    smiles_attn=smiles_attn,
                    protein_emb=protein_emb,
                    protein_attn=protein_attn,
                    device=device,
                    gpu=rank,
                )

                loss = mse_loss(outputs, labels.float())
            total_loss += loss.item()
            y_label.append(labels.squeeze(-1).float().cpu().numpy())
            y_pred.append(outputs.squeeze(-1).float().cpu().numpy())

    y_label = np.concatenate(y_label)
    y_pred = np.concatenate(y_pred)
    _pcc, _scc, _r2, _rmse, _mse, _mae = evaluate(y_label, y_pred)
    return y_pred, y_label, np.array([_pcc, _scc, _r2, _rmse, _mse, _mae, total_loss / max(len(data_loader), 1)])


def run_eval_mse_epoch(model, data_loader, device, rank=0, autocast_dtype=None):
    """
    Fast validation pass for early stopping.
    Returns np.array([mse, avg_loss]).
    """
    model.eval()
    total_loss = 0.0
    mse_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in data_loader:
            smiles_emb, smiles_attn, protein_emb, protein_attn, labels = _move_batch(batch, device)

            with _autocast_context(device, autocast_dtype=autocast_dtype):
                outputs = model(
                    smiles_emb=smiles_emb,
                    smiles_attn=smiles_attn,
                    protein_emb=protein_emb,
                    protein_attn=protein_attn,
                    device=device,
                    gpu=rank,
                )
                loss = mse_loss(outputs, labels.float())

            labels_f = labels.squeeze(-1).float()
            outputs_f = outputs.squeeze(-1).float()
            batch_size = int(labels_f.size(0))
            mse_sum += float(torch.sum((outputs_f - labels_f) ** 2).item())
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size

    denom = max(sample_count, 1)
    return np.array([mse_sum / denom, total_loss / denom], dtype=np.float64)
