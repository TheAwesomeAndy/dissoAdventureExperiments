#!/usr/bin/env python3
"""
Chapter 5 Verification Script
==============================
Verifies that the Chapter 5 scripts are syntactically valid, importable,
and that core components (reservoir, feature extraction, graph construction,
GNN propagation) produce correct outputs on minimal synthetic data.

The full pipeline requires the SHAPE EEG dataset which is not
included in the repository. This script tests the code infrastructure
without external data.

Usage:
    python chapter5Experiments/verify_chapter5.py

Exit code 0 = all checks pass, 1 = at least one check failed.
"""
import sys
import os

# Windows cp1252 portability: scripts print Unicode box-drawing chars (─, ═)
# and read source files containing UTF-8 (µ, ≈, ≥). Without this, they crash
# on default Windows consoles. Python 3.7+ has reconfigure; older silently skip.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass
import importlib.util
import numpy as np

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}  {detail}")


def load_module(name, path):
    """Import a module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    global PASS, FAIL
    base = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("CHAPTER 5 VERIFICATION (infrastructure tests, no external data)")
    print("=" * 70)

    # ── 1. Import test ──
    print("\n--- Script Import ---")
    try:
        mod = load_module("ch5_exp", os.path.join(base,
                          "run_chapter5_experiments.py"))
        check("run_chapter5_experiments.py imports successfully", True)
    except Exception as e:
        check("run_chapter5_experiments.py imports successfully", False,
              str(e))
        print("Cannot continue without successful import.")
        return 1

    # ── 2. LIF Reservoir ──
    print("\n--- LIF Reservoir ---")
    res = mod.LIFReservoir(n_res=64, beta=0.05, threshold=0.5, seed=42)
    check("Reservoir instantiates (64 neurons)", True)
    check("W_in shape correct", res.W_in.shape == (64, 1))
    check("W_rec shape correct", res.W_rec.shape == (64, 64))

    # Test forward pass with synthetic signal
    signal = np.sin(np.linspace(0, 4 * np.pi, 100)) * 2.0
    spikes = res.forward(signal)
    check("Forward pass returns (100, 64) spikes", spikes.shape == (100, 64))
    check("Spikes are binary (0 or 1)",
          set(np.unique(spikes)).issubset({0.0, 1.0}))
    check("Spikes are not all zero", spikes.sum() > 0)
    check("Spikes are sparse (< 50% active)",
          spikes.mean() < 0.5,
          f"mean={spikes.mean():.3f}")

    # ── 3. Feature Extraction ──
    print("\n--- Feature Extraction ---")
    bsc6 = mod.extract_bsc6(spikes)
    check("BSC6 produces 384-dim vector (64 neurons x 6 bins)",
          bsc6.shape == (64 * 6,),
          f"got shape {bsc6.shape}")
    check("BSC6 values are non-negative", np.all(bsc6 >= 0))

    mfr = mod.extract_mfr(spikes)
    check("MFR produces 64-dim vector", mfr.shape == (64,))
    check("MFR values in [0, 1]", np.all(mfr >= 0) and np.all(mfr <= 1))

    # ── 4. Conventional Features ──
    print("\n--- Conventional Features ---")
    # Create minimal multichannel data: (2 samples, 256 timesteps, 34 channels)
    X_test = np.random.randn(2, 256, 34)

    bp = mod.extract_bandpower(X_test, fs=256)
    check("BandPower shape (2, 34, 5)", bp.shape == (2, 34, 5),
          f"got {bp.shape}")
    check("BandPower values non-negative", np.all(bp >= 0))

    hj = mod.extract_hjorth(X_test)
    check("Hjorth shape (2, 34, 3)", hj.shape == (2, 34, 3),
          f"got {hj.shape}")

    # ── 5. Graph Construction ──
    print("\n--- Graph Construction ---")
    positions, ch_names = mod.get_standard_34ch_positions()
    check("34 electrode positions returned", positions.shape == (34, 2))
    check("34 channel names returned", len(ch_names) == 34)

    A_spat = mod.build_spatial_adjacency(positions, k_neighbors=5)
    check("Spatial adjacency shape (34, 34)", A_spat.shape == (34, 34))
    check("Spatial adjacency is symmetric",
          np.allclose(A_spat, A_spat.T))
    check("Spatial adjacency is binary",
          set(np.unique(A_spat)).issubset({0.0, 1.0}))
    check("No self-loops", np.all(np.diag(A_spat) == 0))

    fake_feats = np.random.randn(34, 64)
    A_func = mod.build_functional_adjacency(fake_feats)
    check("Functional adjacency shape (34, 34)", A_func.shape == (34, 34))
    check("Functional adjacency is symmetric",
          np.allclose(A_func, A_func.T, atol=1e-10))

    # ── 6. GNN Propagation ──
    print("\n--- GNN Propagation ---")
    H = np.random.randn(34, 64)

    A_norm = mod.normalize_adjacency(A_spat)
    H_gcn = mod.gcn_propagate(H, A_norm, n_layers=2)
    check("GCN propagation preserves shape (34, 64)",
          H_gcn.shape == (34, 64))

    H_sage = mod.graphsage_propagate(H, A_spat, n_layers=1)
    check("GraphSAGE doubles features per layer",
          H_sage.shape == (34, 128))

    H_gat, attns = mod.gat_propagate(H, A_spat + np.eye(34), n_layers=1)
    check("GAT propagation returns features", H_gat.shape[0] == 34)
    check("GAT returns attention matrices", len(attns) == 1)
    check("Attention rows sum to ~1",
          np.allclose(attns[0].sum(axis=1), 1.0, atol=1e-5))

    # ── 7. Graph Readout ──
    print("\n--- Graph Readout ---")
    g = mod.graph_readout(H, mode='mean')
    check("Mean readout produces 64-dim vector", g.shape == (64,))

    # ── 8. Classification Pipeline ──
    print("\n--- Classification Pipeline (minimal) ---")
    # Create tiny synthetic node features for 10 samples
    N = 20
    X_nodes = np.random.randn(N, 34, 8)
    y = np.array([0] * 10 + [1] * 10)
    subjects = np.array([f"{i:03d}" for i in range(N)])
    try:
        result = mod.run_full_cv(X_nodes, y, subjects, A_spat,
                                 gnn_type='gcn', n_layers=1,
                                 classifier='logreg', n_folds=2)
        check("Full CV pipeline runs without error", True)
        check("CV returns accuracy", 'accuracy' in result)
        check("CV returns predictions",
              len(result['predictions']) == N)
    except Exception as e:
        check("Full CV pipeline runs without error", False, str(e))

    # ── Summary ──
    print("\n" + "=" * 70)
    print(f"CHAPTER 5 VERIFICATION COMPLETE: {PASS} passed, {FAIL} failed")
    print("=" * 70)
    print("\nNote: Full end-to-end verification requires the SHAPE")
    print("EEG dataset. Use --demo mode to test the complete pipeline:")
    print("  python chapter5Experiments/run_chapter5_experiments.py --demo")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
