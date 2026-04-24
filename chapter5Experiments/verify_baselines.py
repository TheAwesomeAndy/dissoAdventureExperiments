#!/usr/bin/env python3
"""
Verification script for sklearn_baselines.py and deprecated/eegnet_gru_lstm_baselines.py.

Tests all core components on synthetic data without requiring the SHAPE dataset.
Validates classifier instantiation, feature extraction, deep model forward passes,
and CV pipeline correctness.

Run:
    python chapter5Experiments/verify_baselines.py
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
import numpy as np

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")


def main():
    global PASS, FAIL
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("VERIFICATION: sklearn_baselines.py + deprecated/eegnet_gru_lstm_baselines.py")
    print("=" * 70)

    # ── 1. Syntax validation ──
    print("\n── Syntax Validation ──")
    for script in ["sklearn_baselines.py", os.path.join("deprecated", "eegnet_gru_lstm_baselines.py")]:
        path = os.path.join(script_dir, script)
        try:
            with open(path, encoding='utf-8') as f:
                compile(f.read(), path, 'exec')
            check(f"{script} parses without syntax errors", True)
        except SyntaxError as e:
            check(f"{script} parses without syntax errors", False, str(e))

    # ── 2. Import sklearn_baselines ──
    print("\n── sklearn_baselines.py Imports ──")
    sys.path.insert(0, script_dir)
    try:
        from sklearn_baselines import (extract_bandpower, extract_hjorth,
                                        CLASSIFIERS, evaluate_classifier)
        check("sklearn_baselines core functions importable", True)
    except ImportError as e:
        check("sklearn_baselines core functions importable", False, str(e))
        return 1

    # ── 3. Feature extraction ──
    print("\n── Conventional Feature Extraction ──")
    np.random.seed(42)
    X_synth = np.random.randn(10, 256, 34) * 10  # (N, T, channels)

    bp = extract_bandpower(X_synth, fs=256)
    check(f"Band power shape (10, 34, 5): {bp.shape}", bp.shape == (10, 34, 5))
    check("Band power non-negative", np.all(bp >= 0))
    check("Band power finite", np.all(np.isfinite(bp)))
    check("Band power has nonzero values", bp.sum() > 0)

    hj = extract_hjorth(X_synth)
    check(f"Hjorth shape (10, 34, 3): {hj.shape}", hj.shape == (10, 34, 3))
    check("Hjorth activity > 0", np.all(hj[:, :, 0] >= 0))
    check("Hjorth mobility > 0", np.all(hj[:, :, 1] >= 0))
    check("Hjorth finite", np.all(np.isfinite(hj)))

    # Combined
    conv = np.concatenate([bp, hj], axis=2)
    features = conv.reshape(10, -1)
    check(f"Flattened features (10, 272): {features.shape}", features.shape == (10, 272))

    # ── 4. All 8 sklearn classifiers instantiate ──
    print("\n── sklearn Classifier Instantiation ──")
    check(f"8 classifiers defined", len(CLASSIFIERS) == 8)
    for name, factory in CLASSIFIERS.items():
        try:
            clf = factory()
            check(f"{name} instantiates", True)
        except Exception as e:
            check(f"{name} instantiates", False, str(e))

    # ── 5. CV pipeline with synthetic data ──
    print("\n── sklearn CV Pipeline ──")
    np.random.seed(42)
    N = 30
    X_cv = np.random.randn(N, 50)
    y_cv = np.repeat([0, 1, 2], 10)
    subj_cv = np.repeat([f"s{i}" for i in range(10)], 3)

    from sklearn.linear_model import LogisticRegression
    accs, f1s = evaluate_classifier(X_cv, y_cv, subj_cv,
                                     lambda: LogisticRegression(C=0.1, max_iter=1000,
                                                                 random_state=42),
                                     n_folds=5)
    check("evaluate_classifier returns arrays", isinstance(accs, np.ndarray))
    check(f"5 fold accuracies returned", len(accs) == 5)
    check("Accuracies in [0, 1]", np.all((accs >= 0) & (accs <= 1)))
    check(f"5 fold F1 scores returned", len(f1s) == 5)

    # ── 6. Import eegnet_gru_lstm_baselines (from deprecated/) ──
    print("\n── deprecated/eegnet_gru_lstm_baselines.py Imports ──")
    try:
        import importlib.util
        _spec = importlib.util.spec_from_file_location(
            "eegnet_gru_lstm_baselines",
            os.path.join(script_dir, "deprecated", "eegnet_gru_lstm_baselines.py"),
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        EEGNetNumpy = _mod.EEGNetNumpy
        GRUClassifier = _mod.GRUClassifier
        LSTMClassifier = _mod.LSTMClassifier
        check("Deep baseline classes importable", True)
    except ImportError as e:
        check("Deep baseline classes importable", False, str(e))
        return 1

    # ── 7. EEGNet ──
    print("\n── EEGNet (NumPy) ──")
    eegnet = EEGNetNumpy(n_channels=34, n_times=256, n_classes=3, seed=42)
    check("EEGNet instantiates", True)

    # Forward pass on single sample
    x_sample = np.random.randn(34, 256)
    logits, feat = eegnet._forward(x_sample)
    check(f"EEGNet logits shape: {logits.shape}", logits.shape == (3,))
    check("EEGNet logits finite", np.all(np.isfinite(logits)))

    # Batch prediction
    X_batch = np.random.randn(5, 34, 256)
    preds = eegnet.predict(X_batch)
    check(f"EEGNet batch predict (5 samples)", len(preds) == 5)
    check("EEGNet predictions in {0,1,2}", set(preds).issubset({0, 1, 2}))

    # Training (short)
    y_train = np.array([0, 1, 2, 0, 1])
    eegnet.fit(X_batch, y_train, epochs=3, lr=0.01)
    check("EEGNet fit runs without error", True)

    # ── 8. GRU ──
    print("\n── GRU Classifier (NumPy) ──")
    gru = GRUClassifier(input_dim=34, hidden_dim=32, n_classes=3, seed=42)
    check("GRU instantiates", True)

    x_seq = np.random.randn(256, 34)
    h = gru._forward_seq(x_seq)
    check(f"GRU hidden state shape (32,)", h.shape == (32,))
    check("GRU hidden state finite", np.all(np.isfinite(h)))

    X_gru = np.random.randn(5, 256, 34)
    preds_gru = gru.predict(X_gru)
    check(f"GRU batch predict (5 samples)", len(preds_gru) == 5)
    check("GRU predictions in {0,1,2}", set(preds_gru).issubset({0, 1, 2}))

    gru.fit(X_gru, y_train, epochs=3)
    check("GRU fit runs without error", True)

    # GRU determinism
    gru1 = GRUClassifier(input_dim=34, hidden_dim=32, seed=42)
    gru2 = GRUClassifier(input_dim=34, hidden_dim=32, seed=42)
    h1 = gru1._forward_seq(x_seq)
    h2 = gru2._forward_seq(x_seq)
    check("GRU deterministic (same seed = same output)", np.allclose(h1, h2))

    # ── 9. LSTM ──
    print("\n── LSTM Classifier (NumPy) ──")
    lstm = LSTMClassifier(input_dim=34, hidden_dim=32, n_classes=3, seed=42)
    check("LSTM instantiates", True)

    h_lstm = lstm._forward_seq(x_seq)
    check(f"LSTM hidden state shape (32,)", h_lstm.shape == (32,))
    check("LSTM hidden state finite", np.all(np.isfinite(h_lstm)))

    X_lstm = np.random.randn(5, 256, 34)
    preds_lstm = lstm.predict(X_lstm)
    check(f"LSTM batch predict (5 samples)", len(preds_lstm) == 5)

    lstm.fit(X_lstm, y_train, epochs=3)
    check("LSTM fit runs without error", True)

    # LSTM determinism
    lstm1 = LSTMClassifier(input_dim=34, hidden_dim=32, seed=42)
    lstm2 = LSTMClassifier(input_dim=34, hidden_dim=32, seed=42)
    h_l1 = lstm1._forward_seq(x_seq)
    h_l2 = lstm2._forward_seq(x_seq)
    check("LSTM deterministic (same seed = same output)", np.allclose(h_l1, h_l2))

    # ── 10. GRU ≠ LSTM ──
    print("\n── Model Distinctness ──")
    check("GRU and LSTM produce different outputs",
          not np.allclose(h, h_lstm),
          "GRU and LSTM should differ")

    # ── 11. Pre-existing deep baseline results ──
    print("\n── Pre-existing Results Validation ──")
    results_path = os.path.join(script_dir, "results", "baseline_deep_results.pkl")
    if os.path.exists(results_path):
        import pickle
        with open(results_path, 'rb') as f:
            existing = pickle.load(f)
        check("Existing results pickle loads", True)
        check("Contains EEGNet results", 'EEGNet' in existing)
        check("Contains GRU results", 'GRU' in existing)
        check("Contains LSTM results", 'LSTM' in existing)

        eegnet_acc = existing['EEGNet']['acc']
        gru_acc = existing['GRU']['acc']
        lstm_acc = existing['LSTM']['acc']
        check(f"EEGNet acc ~72% (actual: {eegnet_acc*100:.1f}%)",
              abs(eegnet_acc - 0.72) < 0.05)
        check(f"GRU acc ~60% (actual: {gru_acc*100:.1f}%)",
              abs(gru_acc - 0.60) < 0.05)
        check(f"LSTM acc ~58% (actual: {lstm_acc*100:.1f}%)",
              abs(lstm_acc - 0.58) < 0.05)
        check("EEGNet > GRU > LSTM ordering", eegnet_acc > gru_acc > lstm_acc)
        check("10 fold accuracies per model",
              all(len(existing[m]['fold_accs']) == 10 for m in ['EEGNet', 'GRU', 'LSTM']))
    else:
        check("Existing results pickle (optional — generated by full run)", True)

    # ── Summary ──
    print("\n" + "=" * 70)
    total = PASS + FAIL
    print(f"RESULT: {PASS}/{total} PASS, {FAIL}/{total} FAIL")
    if FAIL == 0:
        print("All tests passed.")
    print("=" * 70)
    return FAIL


if __name__ == "__main__":
    sys.exit(main())
