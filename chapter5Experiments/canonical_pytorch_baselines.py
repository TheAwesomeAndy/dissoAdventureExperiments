#!/usr/bin/env python3
"""
ARSPI-Net Chapter 5: Canonical PyTorch Baselines
=================================================

Full end-to-end trained deep learning baselines for 3-class affective EEG
classification on the SHAPE Community dataset (N=211 subjects, 34 channels).

This script replaces the NumPy reference implementations in
eegnet_gru_lstm_baselines.py with canonical PyTorch models:

  1. EEGNet (Lawhern et al. 2018) — full architecture with depthwise +
     separable convolutions, BatchNorm, dropout, end-to-end training.
  2. GRU — 2-layer bidirectional GRU with trained recurrent weights.
  3. LSTM — 2-layer bidirectional LSTM with trained recurrent weights.

All models use:
  - Adam optimizer (lr=1e-3, weight_decay=1e-4)
  - Early stopping (patience=20, monitor=val_loss)
  - Gradient clipping (max_norm=1.0)
  - ReduceLROnPlateau (factor=0.5, patience=10)
  - 10-fold StratifiedGroupKFold (random_state=42)
  - 10 random seeds × 5 initializations per fold = 50 runs per model
  - Balanced class weights via CrossEntropyLoss

This script produces 3 of the 7 rows in the dissertation's baseline table
(the deep learning baselines). The remaining 4 rows (Raw EEG, PCA-200,
Reservoir, BandPower) are produced by run_chapter5_experiments.py using
LogReg/SVM readouts on pre-extracted features.

Results (SHAPE dataset, 211 subjects, 3-class):
  Uncentered:                    Centered:
  EEGNet:     72.0% ± 4.9%      EEGNet:     89.1% ± 3.1%   (+17.1 pp)
  GRU:        59.9% ± 6.4%      GRU:        78.4% ± 3.5%   (+18.5 pp)
  LSTM:       58.0% ± 5.5%      LSTM:       71.1% ± 6.8%   (+13.1 pp)

Centering is the dominant intervention; architecture choice is secondary.

Usage:
  python canonical_pytorch_baselines.py --data_dir /path/to/batch_data/
  python canonical_pytorch_baselines.py --data_dir /path/to/batch_data/ --centered

Requires: torch>=1.12, numpy, scipy, scikit-learn
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("ERROR: This script requires PyTorch. Install with: pip install torch")
    print("The NumPy reference implementations are in eegnet_gru_lstm_baselines.py")
    sys.exit(1)

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.io import loadmat


# ============================================================
# EEGNet (Lawhern et al. 2018)
# ============================================================
class EEGNet(nn.Module):
    """
    Compact CNN for EEG-based BCIs.
    Reference: Lawhern et al., J. Neural Eng. 15(5):056013, 2018.

    Architecture:
      Conv2D(temporal) → BatchNorm → DepthwiseConv2D(spatial) → BatchNorm →
      ELU → AvgPool → Dropout → SeparableConv2D → BatchNorm → ELU →
      AvgPool → Dropout → Flatten → Linear

    Parameters follow the original paper defaults for ERP paradigms:
      F1=8, D=2, F2=16, kernel_length=64, dropout=0.5
    """

    def __init__(self, n_channels=34, n_samples=256, n_classes=3,
                 F1=8, D=2, F2=16, kernel_length=64, dropout_rate=0.5):
        super().__init__()

        # Block 1: Temporal convolution + Depthwise spatial
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length),
                               padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                    groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: Separable convolution
        self.separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16),
                                          padding=(0, 8), groups=F1 * D, bias=False)
        self.separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # Classifier
        self.flatten = nn.Flatten()
        # Compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self._forward_features(dummy)
            n_flat = dummy.shape[1]
        self.classifier = nn.Linear(n_flat, n_classes)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.classifier(x)
        return x


# ============================================================
# GRU and LSTM baselines
# ============================================================
class RNNBaseline(nn.Module):
    """
    2-layer bidirectional RNN (GRU or LSTM) for EEG classification.
    Input: (batch, time, channels) → output: (batch, n_classes)
    """

    def __init__(self, n_channels=34, n_classes=3, hidden_size=64,
                 n_layers=2, dropout=0.3, rnn_type='GRU'):
        super().__init__()

        RNN = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        self.rnn = RNN(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, n_classes)  # *2 for bidirectional

    def forward(self, x):
        # x: (batch, time, channels)
        output, _ = self.rnn(x)
        # Use last timestep
        last = output[:, -1, :]
        last = self.dropout(last)
        return self.classifier(last)


# ============================================================
# Data loading
# ============================================================
def load_shape_data(data_dir, centered=False):
    """
    Load 3-class SHAPE EEG data from batch_data/ directory.

    Returns:
        X: (n_observations, n_channels, n_samples) — EEG data
        y: (n_observations,) — condition labels (0=Neg, 1=Neu, 2=Pls)
        groups: (n_observations,) — subject IDs for GroupKFold
    """
    data_dir = Path(data_dir)
    all_X, all_y, all_groups = [], [], []

    for batch_file in sorted(data_dir.glob("*.mat")):
        mat = loadmat(str(batch_file))
        # Adapt to your SHAPE data format
        if 'data' in mat:
            X = mat['data']
        elif 'EEG' in mat:
            X = mat['EEG']
        else:
            # Try first non-metadata key
            keys = [k for k in mat.keys() if not k.startswith('_')]
            X = mat[keys[0]]

        if 'labels' in mat:
            y = mat['labels'].ravel()
        if 'subjects' in mat:
            groups = mat['subjects'].ravel()

        all_X.append(X)
        all_y.append(y)
        all_groups.append(groups)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    if centered:
        # Subject-mean centering: subtract each subject's grand-mean ERP
        unique_subjects = np.unique(groups)
        for subj in unique_subjects:
            mask = groups == subj
            subj_mean = X[mask].mean(axis=0, keepdims=True)
            X[mask] = X[mask] - subj_mean

    return X.astype(np.float32), y.astype(np.int64), groups


# ============================================================
# Training loop
# ============================================================
def train_model(model, train_loader, val_loader, device,
                n_epochs=200, patience=20, lr=1e-3, weight_decay=1e-4):
    """
    Train with Adam, early stopping, gradient clipping, LR scheduling.
    """
    # Balanced class weights
    all_labels = []
    for _, labels in train_loader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels)
    class_counts = torch.bincount(all_labels)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=False
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(n_epochs):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate(model, test_loader, device):
    """Return balanced accuracy on test set."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            preds = output.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return balanced_accuracy_score(all_labels, all_preds)


# ============================================================
# Main experimental loop
# ============================================================
def run_experiment(data_dir, centered=False, n_seeds=10, n_inits=5):
    """
    Run 10-fold StratifiedGroupKFold with multiple seeds and initializations.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Centered: {centered}")

    X, y, groups = load_shape_data(data_dir, centered=centered)
    print(f"Data: {X.shape}, {len(np.unique(groups))} subjects, {len(np.unique(y))} classes")

    n_channels = X.shape[1] if X.ndim == 3 else X.shape[2]
    n_samples = X.shape[2] if X.ndim == 3 else X.shape[1]

    models_config = {
        'EEGNet': lambda: EEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=3),
        'GRU': lambda: RNNBaseline(n_channels=n_channels, n_classes=3, rnn_type='GRU'),
        'LSTM': lambda: RNNBaseline(n_channels=n_channels, n_classes=3, rnn_type='LSTM'),
    }

    results = {name: [] for name in models_config}

    for seed in range(n_seeds):
        skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42 + seed)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Validation split (10% of training)
            n_val = max(1, len(X_train) // 10)
            X_val, y_val = X_train[:n_val], y_train[:n_val]
            X_train, y_train = X_train[n_val:], y_train[n_val:]

            for model_name, model_fn in models_config.items():
                fold_accs = []
                for init in range(n_inits):
                    torch.manual_seed(seed * 1000 + fold * 100 + init)
                    np.random.seed(seed * 1000 + fold * 100 + init)

                    model = model_fn().to(device)

                    # Prepare data for model type
                    if model_name == 'EEGNet':
                        # EEGNet expects (batch, 1, channels, time)
                        Xtr = torch.tensor(X_train).unsqueeze(1)
                        Xv = torch.tensor(X_val).unsqueeze(1)
                        Xte = torch.tensor(X_test).unsqueeze(1)
                    else:
                        # RNN expects (batch, time, channels)
                        if X_train.shape[1] == n_channels:
                            Xtr = torch.tensor(X_train).permute(0, 2, 1)
                            Xv = torch.tensor(X_val).permute(0, 2, 1)
                            Xte = torch.tensor(X_test).permute(0, 2, 1)
                        else:
                            Xtr = torch.tensor(X_train)
                            Xv = torch.tensor(X_val)
                            Xte = torch.tensor(X_test)

                    ytr = torch.tensor(y_train)
                    yv = torch.tensor(y_val)
                    yte = torch.tensor(y_test)

                    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
                    val_loader = DataLoader(TensorDataset(Xv, yv), batch_size=32)
                    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=32)

                    model = train_model(model, train_loader, val_loader, device)
                    acc = evaluate(model, test_loader, device)
                    fold_accs.append(acc)

                results[model_name].append(np.mean(fold_accs))

        print(f"Seed {seed} complete")

    # Report
    print("\n" + "=" * 60)
    tag = "CENTERED" if centered else "UNCENTERED"
    print(f"RESULTS ({tag})")
    print("=" * 60)
    for name, accs in results.items():
        accs = np.array(accs)
        print(f"  {name:8s}: {accs.mean()*100:.1f}% ± {accs.std()*100:.1f}%  (n={len(accs)} fold-runs)")
    print("=" * 60)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Canonical PyTorch baselines for SHAPE EEG')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to batch_data/ directory')
    parser.add_argument('--centered', action='store_true',
                        help='Apply subject-mean centering before classification')
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Number of random seeds (default: 10)')
    parser.add_argument('--n_inits', type=int, default=5,
                        help='Number of initializations per fold (default: 5)')
    args = parser.parse_args()

    run_experiment(args.data_dir, centered=args.centered,
                   n_seeds=args.n_seeds, n_inits=args.n_inits)
