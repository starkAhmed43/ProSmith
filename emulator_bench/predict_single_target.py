"""
predict_single_target.py — ProSmith inference script.

Loads a trained MM_TN checkpoint and predicts on a feature pkl.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "code"))

from training.utils.modules import MM_TN, MM_TNConfig  # noqa: E402


def _build_model(num_hidden_layers=6):
    config = MM_TNConfig.from_dict({
        "s_hidden_size": 600,
        "p_hidden_size": 1280,
        "hidden_size": 768,
        "max_seq_len": 1276,
        "num_hidden_layers": num_hidden_layers,
        "binary_task": False,
    })
    return MM_TN(config)


def predict_from_pkl(input_pkl, ckpt_path, out_csv, device="cuda:0", num_hidden_layers=6, batch_size=32):
    from train_single_target_tvt import ProSmithDataset  # local import

    ds = ProSmithDataset(input_pkl)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ProSmithDataset.collate_fn,
        num_workers=2,
    )

    model = _build_model(num_hidden_layers=num_hidden_layers)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            smiles_emb, smiles_attn, protein_emb, protein_attn, _ = [b.to(device) for b in batch]
            out = model(
                smiles_emb=smiles_emb,
                smiles_attn=smiles_attn,
                protein_emb=protein_emb,
                protein_attn=protein_attn,
                device=device,
                gpu=0,
            )
            all_preds.append(out.squeeze(-1).float().cpu().numpy())

    preds = np.concatenate(all_preds)
    with open(input_pkl, "rb") as f:
        payload = pickle.load(f)
    out_df = pd.DataFrame({"pred": preds})
    out_df.to_csv(out_csv, float_format="%.6f", index=False)
    print(f"Saved predictions to: {out_csv}  ({len(preds)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Predict with trained MM_TN checkpoint.")
    parser.add_argument("--input_pkl", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--out_csv", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--num_hidden_layers", default=6, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    args = parser.parse_args()

    predict_from_pkl(args.input_pkl, args.ckpt_path, args.out_csv, args.device, args.num_hidden_layers, args.batch_size)


if __name__ == "__main__":
    main()
