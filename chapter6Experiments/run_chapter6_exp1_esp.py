#!/usr/bin/env python3
"""
run_chapter6_exp1_esp.py
========================
ARSPI-Net Chapter 6, Experiment 6.1:
Echo State Property and Driven Stability Verification

Validates that the LIF reservoir (N=256, β=0.05, M_th=0.5) operates in a
stable driven regime under real clinical EEG from the SHAPE dataset.

Two analyses:
  6.1a — ESP Convergence: Drive the same reservoir from two widely separated
         initial conditions with identical EEG input. Verify trajectory convergence.
  6.1b — Driven Lyapunov Exponent: Benettin renormalization to compute λ₁
         across 211 subjects × 4 affective subcategories × 5 channels.

Results:
  - λ₁ = -0.0536 ± 0.0001 (100% negative across 4,220 measurements)
  - The reservoir is uniformly contracting under real EEG drive
  - ESP is empirically verified: state trajectories are input-determined

Usage:
    python run_chapter6_exp1_esp.py --category-dirs categoriesbatch1 ... --output-dir results/

Author: Andrew (ARSPI-Net Dissertation)
"""

import numpy as np
import os, re, pickle, time, argparse
from collections import defaultdict

# ============================================================
# Reservoir Configuration (Chapter 3/4 validated)
# ============================================================
N_RES = 256
BETA = 0.05
M_TH = 0.5
P_CONNECT = 0.1
W_SCALE_IN = 0.3
W_SCALE_REC = 0.05
EXCLUDED_SUBJECTS = {127}

# Benettin parameters
DELTA_0 = 1e-8
T_RENORM = 50

FILE_PATTERN = re.compile(
    r'SHAPE_Community_(\d+)_IAPS(Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC\.txt'
)


class LIFReservoir:
    """Leaky Integrate-and-Fire reservoir with fixed random weights."""
    
    def __init__(self, n_input, n_res=N_RES, seed=42):
        rng = np.random.RandomState(seed)
        self.n_res = n_res
        self.W_in = rng.randn(n_res, n_input) * W_SCALE_IN
        mask = (rng.rand(n_res, n_res) < P_CONNECT).astype(float)
        np.fill_diagonal(mask, 0)
        self.W_rec = rng.randn(n_res, n_res) * W_SCALE_REC * mask
    
    def step(self, m_prev, s_prev, u_t):
        """Single time step. Returns (m_new, s_new)."""
        I_in = self.W_in @ u_t
        I_rec = self.W_rec @ s_prev
        m_new = (1 - BETA) * m_prev * (1 - s_prev) + I_in + I_rec
        s_new = (m_new >= M_TH).astype(float)
        return m_new, s_new
    
    def run(self, u, M0=None):
        """Full forward pass. u: (T, n_input). Returns M (T, n_res), S (T, n_res)."""
        T, n_in = u.shape
        M = np.zeros((T, self.n_res))
        S = np.zeros((T, self.n_res), dtype=np.int8)
        m = M0.copy() if M0 is not None else np.zeros(self.n_res)
        s = np.zeros(self.n_res)
        for t in range(T):
            m, s = self.step(m, s, u[t])
            M[t] = m
            S[t] = s.astype(np.int8)
        return M, S


def load_subcategory_files(category_dirs):
    """Load file index: {(subject_id, category): filepath}."""
    files = {}
    for cdir in category_dirs:
        if not os.path.isdir(cdir):
            continue
        for f in os.listdir(cdir):
            m = FILE_PATTERN.match(f)
            if m:
                sid = int(m.group(1))
                cat = m.group(3)
                if sid not in EXCLUDED_SUBJECTS:
                    files[(sid, cat)] = os.path.join(cdir, f)
    return files


def esp_convergence(reservoir, u, delta_scale=2.0, seed=999):
    """Test ESP by running from two initial conditions.
    
    Returns dict with convergence curve and statistics.
    """
    T = u.shape[0]
    M1, S1 = reservoir.run(u, M0=np.zeros(reservoir.n_res))
    
    rng = np.random.RandomState(seed)
    M0_rand = rng.rand(reservoir.n_res) * M_TH * delta_scale
    M2, S2 = reservoir.run(u, M0=M0_rand)
    
    state_diff = np.sqrt(np.mean((M1 - M2)**2, axis=1))
    initial_diff = state_diff[0]
    
    converged = False
    conv_time = -1
    if initial_diff > 0:
        mask = state_diff < 0.01 * initial_diff
        if mask.any():
            conv_time = int(np.argmax(mask))
            converged = True
    
    return {
        'state_diff_curve': state_diff[::10],  # subsample
        'initial_diff': float(initial_diff),
        'final_diff': float(state_diff[-1]),
        'conv_time': conv_time,
        'converged': converged,
    }


def driven_lyapunov(reservoir, u, delta_0=DELTA_0, t_renorm=T_RENORM, seed=42):
    """Compute driven Lyapunov exponent via Benettin renormalization.
    
    Returns float: estimated λ₁.
    """
    T = u.shape[0]
    rng = np.random.RandomState(seed)
    e_hat = rng.randn(reservoir.n_res)
    e_hat /= np.linalg.norm(e_hat)
    
    m_ref = np.zeros(reservoir.n_res)
    s_ref = np.zeros(reservoir.n_res)
    m_pert = m_ref + delta_0 * e_hat
    s_pert = np.zeros(reservoir.n_res)
    
    log_stretches = []
    
    for t in range(T):
        m_ref, s_ref = reservoir.step(m_ref, s_ref, u[t])
        m_pert, s_pert = reservoir.step(m_pert, s_pert, u[t])
        
        if (t + 1) % t_renorm == 0:
            diff = m_pert - m_ref
            dist = np.linalg.norm(diff)
            if dist > 0:
                log_stretches.append(np.log(dist / delta_0))
                m_pert = m_ref + delta_0 * (diff / dist)
                s_pert = s_ref.copy()
    
    if log_stretches:
        return np.mean(log_stretches) / t_renorm
    return 0.0


def main():
    parser = argparse.ArgumentParser(description='Chapter 6 Experiment 6.1: ESP Verification')
    parser.add_argument('--category-dirs', nargs='+', required=True)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--channels', nargs='+', type=int, default=[0, 8, 16, 24, 33])
    parser.add_argument('--esp-subjects', type=int, default=30,
                        help='Number of subjects for detailed ESP convergence')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    
    print("=" * 70)
    print("EXPERIMENT 6.1: DRIVEN STABILITY / ESP VERIFICATION")
    print("=" * 70)
    
    # Load data
    all_files = load_subcategory_files(args.category_dirs)
    subjects = sorted(set(s for s, c in all_files.keys()))
    categories = ['Threat', 'Mutilation', 'Cute', 'Erotic']
    print(f"Loaded {len(all_files)} files, {len(subjects)} subjects")
    
    # Build reservoir
    reservoir = LIFReservoir(n_input=1, seed=args.seed)
    
    # --- 6.1a: ESP Convergence ---
    print(f"\n--- 6.1a: ESP Convergence ({args.esp_subjects} subjects) ---")
    convergence_results = []
    for si, sid in enumerate(subjects[:args.esp_subjects]):
        if si % 10 == 0:
            print(f"  Subject {sid} ({si+1}/{args.esp_subjects})...")
        
        key = (sid, 'Threat')
        if key not in all_files:
            continue
        eeg = np.loadtxt(all_files[key])
        
        for ch in args.channels:
            u = eeg[:, ch:ch+1]
            u = (u - u.mean()) / (u.std() + 1e-10)
            result = esp_convergence(reservoir, u)
            result['subject'] = sid
            result['channel'] = ch
            convergence_results.append(result)
    
    n_conv = sum(1 for r in convergence_results if r['converged'])
    print(f"  Converged (1% threshold): {n_conv}/{len(convergence_results)}")
    print(f"  Mean final diff: {np.mean([r['final_diff'] for r in convergence_results]):.4f}")
    
    # --- 6.1b: Driven Lyapunov ---
    print(f"\n--- 6.1b: Driven Lyapunov ({len(subjects)} subjects × {len(categories)} categories) ---")
    lyapunov_results = []
    for si, sid in enumerate(subjects):
        if si % 50 == 0:
            print(f"  Subject {sid} ({si+1}/{len(subjects)})... {time.time()-t0:.0f}s")
        
        for cat in categories:
            key = (sid, cat)
            if key not in all_files:
                continue
            eeg = np.loadtxt(all_files[key])
            
            for ch in args.channels:
                u = eeg[:, ch:ch+1]
                u = (u - u.mean()) / (u.std() + 1e-10)
                lam1 = driven_lyapunov(reservoir, u, seed=args.seed)
                lyapunov_results.append({
                    'subject': sid, 'channel': ch, 'category': cat,
                    'lambda1': lam1,
                })
    
    # Report
    lam_vals = np.array([r['lambda1'] for r in lyapunov_results])
    print(f"\n  Results ({len(lam_vals)} measurements):")
    print(f"    Mean λ₁:  {lam_vals.mean():.6f} ± {lam_vals.std():.6f}")
    print(f"    Range:    [{lam_vals.min():.6f}, {lam_vals.max():.6f}]")
    print(f"    λ₁ < 0:   {(lam_vals < 0).sum()}/{len(lam_vals)} ({100*(lam_vals<0).mean():.1f}%)")
    
    for cat in categories:
        cat_lam = [r['lambda1'] for r in lyapunov_results if r['category'] == cat]
        print(f"    {cat:12s}: {np.mean(cat_lam):.6f} ± {np.std(cat_lam):.6f}")
    
    # Save
    output_path = os.path.join(args.output_dir, 'ch6_exp1_esp.pkl')
    pickle.dump({
        'convergence_results': convergence_results,
        'lyapunov_results': lyapunov_results,
        'subjects': subjects,
        'categories': categories,
        'channels': args.channels,
    }, open(output_path, 'wb'))
    
    print(f"\nResults saved to {output_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
