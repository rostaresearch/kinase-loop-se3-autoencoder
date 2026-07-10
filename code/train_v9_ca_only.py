"""Train molearn Small_AutoEncoder on the v9 Cα-only dataset.

Identical to ``train_v4_foldingnet.py`` except for the atom selection
(``["CA"]`` instead of ``["CA","C","N","CB","O"]``) — matches the
meyresearch/BRAF published workflow's
``AutoencoderWorkflow.prepare_data(atom_selection=['CA'])``.

Usage::

    python train_v9_ca_only.py \\
        --combined-pdb /tmp/v9_ca/combined_v9_ca.pdb \\
        --out         /tmp/v9_ca/run \\
        --max-cycles 8 --epochs-per-cycle 32 --batch-size 8
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-pdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--max-cycles", type=int, default=8)
    ap.add_argument("--epochs-per-cycle", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--valid-ratio", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "xbb_foldingnet_checkpoints").mkdir(parents=True, exist_ok=True)

    from molearn.data import PDBData
    from molearn.trainers import Trainer
    from molearn.models.small_foldingnet import Small_AutoEncoder

    data = PDBData()
    data.import_pdb(filename=str(args.combined_pdb))
    data.fix_terminal()
    data.atomselect(atoms=["CA"])     # ← only change vs train_v4
    data.prepare_dataset()
    print(data._mol)

    n = data.dataset.shape[0]
    idx = np.random.RandomState(seed=args.seed).permutation(n)
    n_test = int(n * args.valid_ratio)
    test_idx = idx[:n_test]; train_idx = idx[n_test:]
    np.savetxt(out / "train_idx.txt", train_idx, fmt="%d")
    np.savetxt(out / "test_idx.txt", test_idx, fmt="%d")
    print(f"Train: {len(train_idx)}  Test: {len(test_idx)}")

    import copy as _copy
    data_train = _copy.copy(data)
    data_train.dataset = data.dataset[train_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    trainer = Trainer(device=device)
    trainer.set_data(data_train, batch_size=args.batch_size,
                     validation_split=0.1, manual_seed=args.seed)
    n_atoms = data.dataset.shape[1]
    print(f"out_points (n_atoms) = {n_atoms}")
    trainer.set_autoencoder(Small_AutoEncoder, out_points=n_atoms)
    trainer.prepare_optimiser()

    log_folder = out / "xbb_foldingnet_checkpoints"
    total_epochs = args.max_cycles * args.epochs_per_cycle
    trainer.run(epochs=total_epochs,
                log_filename="log_file.dat",
                log_folder=str(log_folder),
                checkpoint_folder=str(out))
    try:
        best = float(trainer.best)
    except AttributeError:
        best = float("nan")
    try:
        ep = int(trainer.epoch)
    except AttributeError:
        ep = total_epochs

    with (out / "final.pkl").open("wb") as f:
        pickle.dump({"best": best, "epochs": ep}, f)
    print(f"Done. Best={best}  Epochs={ep}")


if __name__ == "__main__":
    main()
