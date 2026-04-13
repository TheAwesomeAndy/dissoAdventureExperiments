#!/usr/bin/env python3
"""
ARSPI-Net Chapter 5: Deep Learning Baselines — NumPy REFERENCE IMPLEMENTATIONS
================================================================================

*** IMPORTANT: THESE ARE SIMPLIFIED REFERENCE IMPLEMENTATIONS ***

These models are implemented from scratch in NumPy without PyTorch/TensorFlow:
  - EEGNet: Only the fully-connected output layer is trained. The temporal
    and spatial convolution weights are fixed random projections.
  - GRU/LSTM: Recurrent weights are fixed random — only the output layer
    is trained. These are effectively untrained gated reservoirs, NOT
    canonical trained RNNs.

The reported numbers (EEGNet 72.0%, GRU 59.9%, LSTM 58.0%) are FROM THESE
simplified implementations and are NOT directly comparable to canonical
literature implementations.

*** For the dissertation's final baseline table, use: ***
    canonical_pytorch_baselines.py
which implements full end-to-end PyTorch training with:
  - EEGNet per Lawhern et al. 2018
  - 2-layer bidirectional GRU/LSTM with trained recurrent weights
  - Adam, early stopping, gradient clipping, ReduceLROnPlateau
  - 10-fold StratifiedGroupKFold, 10 seeds × 5 initializations

This file is retained for reproducibility of the original exploration.

Original results (SHAPE dataset, 211 subjects, 3-class, UNCENTERED):
  EEGNet (NumPy, FC-only):  72.0% ± 4.9% (MCC 0.585)
  GRU (NumPy, fixed RNN):   59.9% ± 6.4% (MCC 0.406)
  LSTM (NumPy, fixed RNN):  58.0% ± 5.5% (MCC 0.377)

Canonical PyTorch results (canonical_pytorch_baselines.py):
  Uncentered → Centered:
  EEGNet: 72.0% → 89.1%  (+17.1 pp)
  GRU:    59.9% → 78.4%  (+18.5 pp)
  LSTM:   58.0% → 71.1%  (+13.1 pp)

Usage:
  python eegnet_gru_lstm_baselines.py --data_dir /path/to/batch_data/

Requires: numpy, scipy, scikit-learn
  (All models implemented from scratch in NumPy — no PyTorch/TensorFlow)
"""
