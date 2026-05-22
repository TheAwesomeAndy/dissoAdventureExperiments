#!/usr/bin/env python3
"""
graph_diffusion_oversmoothing.py
================================================================
Graph-diffusion over-smoothing analysis for the ARSPI-Net graph layer.

Quantifies WHY message-passing underperforms the non-propagated embedding on
the SHAPE electrode graph (the regime boundary characterized in exp03 /
run_chapter5_experiments.py). Repeated GCN propagation is diffusion on the
normalized graph Laplacian; on a small dense electrode graph the node features
homogenize faster than relational signal accumulates.

Measured vs. propagation depth K, on the real reservoir node features:
  - Dirichlet energy (canonical over-smoothing measure) -> collapses
  - mean pairwise cosine similarity                      -> rises toward uniformity

The over-smoothing onset (steepest change at K=1..2) coincides with the
propagation depth at which exp03 classification accuracy falls fastest. This
is reported as a MECHANISTIC CORRELATE of the regime boundary, not a causal proof.

Operators (LIF reservoir, BSC6, functional adjacency, GCN propagation) are
identical to run_chapter5_experiments.py.

Inputs:
  shape_features_211.pkl  with key 'X_ds' : (N, T, 34) preprocessed EEG
  (gitignored; requires dataset access)

Output:
  pictures/chGraphNeuralNetworks/exp03b_diffusion_overlay.pdf
  diffusion_proxy.npz  (Ks, cos, dir, n_used)

Usage:
  python graph_diffusion_oversmoothing.py --pkl /path/to/shape_features_211.pkl --n 633

Author: Andrew Lane | Stony Brook University ECE
================================================================
"""
import argparse, pickle, time, os
import numpy as np
from sklearn.decomposition import PCA

# repo root = parent of this script's directory (chapter5Experiments/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------
# Reservoir + coding operators (verbatim from run_chapter5_experiments.py)
# ----------------------------------------------------------------
class LIFReservoir:
    """Leaky integrate-and-fire reservoir, single-channel input. Fixed weights."""
    def __init__(self, n_res=256, beta=0.05, threshold=0.5, seed=42):
        rng = np.random.RandomState(seed)
        limit_in = np.sqrt(6.0 / (1 + n_res))
        self.W_in = rng.uniform(-limit_in, limit_in, (n_res, 1))
        limit_rec = np.sqrt(6.0 / (n_res + n_res))
        self.W_rec = rng.uniform(-limit_rec, limit_rec, (n_res, n_res))
        eig_max = np.abs(np.linalg.eigvals(self.W_rec)).max()
        if eig_max > 0:
            self.W_rec *= 0.9 / eig_max
        self.beta, self.threshold, self.n_res = beta, threshold, n_res

    def forward(self, x):
        T = len(x)
        mem = np.zeros(self.n_res)
        spk_prev = np.zeros(self.n_res)
        spikes = np.zeros((T, self.n_res))
        for t in range(T):
            I_in = self.W_in[:, 0] * x[t]
            I_rec = self.W_rec @ spk_prev
            mem = (1.0 - self.beta) * mem * (1.0 - spk_prev) + I_in + I_rec
            spk = (mem >= self.threshold).astype(float)
            mem = np.maximum(mem - spk * self.threshold, 0.0)
            spikes[t] = spk
            spk_prev = spk
        return spikes


def extract_bsc6(spikes):
    """Binned spike counts, 6 temporal bins."""
    T_w, _ = spikes.shape
    bs = T_w // 6
    return np.concatenate([spikes[b * bs:(b + 1) * bs].sum(axis=0) for b in range(6)])


# ----------------------------------------------------------------
# Graph operators (verbatim from run_chapter5_experiments.py)
# ----------------------------------------------------------------
def build_functional_adjacency(node_features, threshold_percentile=75):
    corr = np.nan_to_num(np.corrcoef(node_features), nan=0.0)
    np.fill_diagonal(corr, 0.0)
    pos = corr[corr > 0]
    if pos.size == 0:
        return None
    thresh = np.percentile(pos, threshold_percentile)
    return (corr >= thresh).astype(float)


def normalize_adjacency(A):
    A_tilde = A + np.eye(A.shape[0])
    D = np.diag(A_tilde.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-12))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt


# ----------------------------------------------------------------
# Over-smoothing proxies
# ----------------------------------------------------------------
def mean_pairwise_cosine(H):
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    S = Hn @ Hn.T
    iu = np.triu_indices(H.shape[0], k=1)
    return S[iu].mean()


def dirichlet_energy(H, A_norm):
    diff2 = ((H[:, None, :] - H[None, :, :]) ** 2).sum(axis=2)
    return 0.5 * np.sum(A_norm * diff2) / (np.sum(H ** 2) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='shape_features_211.pkl')
    ap.add_argument('--n', type=int, default=633, help='observations to use')
    ap.add_argument('--max_k', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='pictures/chGraphNeuralNetworks/exp03b_diffusion_overlay.pdf')
    args = ap.parse_args()

    # resolve output relative to repo root unless an absolute path is given
    if not os.path.isabs(args.out):
        args.out = os.path.join(_REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    d = pickle.load(open(args.pkl, 'rb'))
    X_ds = d['X_ds']                      # (N, T, 34)
    N, T, C = X_ds.shape
    print(f"raw X_ds: {X_ds.shape}", flush=True)

    rng = np.random.RandomState(args.seed)
    idx = rng.choice(N, size=min(args.n, N), replace=False)
    reservoirs = [LIFReservoir(n_res=256, seed=42 + ch * 17) for ch in range(C)]

    t0 = time.time()
    bsc = np.zeros((len(idx), C, 6 * 256))
    for j, oi in enumerate(idx):
        for ch in range(C):
            bsc[j, ch] = extract_bsc6(reservoirs[ch].forward(X_ds[oi, :, ch]))
        if (j + 1) % 50 == 0:
            print(f"  reservoir {j+1}/{len(idx)} ({time.time()-t0:.0f}s)", flush=True)

    n_comp = min(64, len(idx) - 1, bsc.shape[2])
    node = np.zeros((len(idx), C, n_comp))
    for ch in range(C):
        node[:, ch, :] = PCA(n_components=n_comp).fit_transform(bsc[:, ch, :])

    Ks = list(range(0, args.max_k + 1))
    cos_by_K = {k: [] for k in Ks}
    dir_by_K = {k: [] for k in Ks}
    used = 0
    for j in range(len(idx)):
        A = build_functional_adjacency(node[j], 75)
        if A is None or A.sum() == 0:
            continue
        A_norm = normalize_adjacency(A)
        H = node[j].copy()
        for k in Ks:
            cos_by_K[k].append(mean_pairwise_cosine(H))
            dir_by_K[k].append(dirichlet_energy(H, A_norm))
            H = A_norm @ H
        used += 1

    cos = np.array([np.mean(cos_by_K[k]) for k in Ks])
    dr = np.array([np.mean(dir_by_K[k]) for k in Ks])
    print(f"\nvalid graphs: {used}/{len(idx)}")
    print("K | cosine | dirichlet")
    for k in Ks:
        print(f"{k} | {cos[k]:.4f} | {dr[k]:.5f}")
    drop = 100 * (dr[2] - dr[0]) / abs(dr[0])
    print(f"\nDirichlet energy K0->K2: {dr[0]:.5f} -> {dr[2]:.5f} ({drop:.0f}%)")

    np.savez('diffusion_proxy.npz', Ks=np.array(Ks), cos=cos, dir=dr, n_used=used)
    _plot(Ks, cos, dr, used, args.out)
    print(f"saved figure -> {args.out}")


def _plot(Ks, cos, dr, n, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    NAVY, ACCENT, TEAL, GREY = '#1A2A4F', '#B5651D', '#1C7293', '#8A94A6'
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 200, 'savefig.dpi': 200,
                         'font.family': 'DejaVu Sans'})
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    l1, = ax.plot(Ks, dr, 'o-', color=NAVY, lw=2.4, ms=6,
                  label='Dirichlet energy (node-feature spread)')
    ax.set_xlabel('GCN propagation depth  $K$', fontsize=13)
    ax.set_ylabel('Dirichlet energy', color=NAVY, fontsize=13)
    ax.tick_params(axis='y', labelcolor=NAVY)
    ax.set_ylim(0, max(dr) * 1.08)
    ax.set_xticks(Ks)
    ax2 = ax.twinx()
    l2, = ax2.plot(Ks, cos, 's--', color=ACCENT, lw=2.4, ms=6,
                   label='Mean pairwise cosine similarity')
    ax2.set_ylabel('Mean pairwise cosine similarity', color=ACCENT, fontsize=13)
    ax2.tick_params(axis='y', labelcolor=ACCENT)
    ax2.set_ylim(0, max(cos) * 1.25)
    ax.axvspan(0.0, 2.0, color=TEAL, alpha=0.08)
    ax.annotate('accuracy in exp03\nfalls fastest here', xy=(1.0, dr[1]),
                xytext=(3.1, max(dr) * 0.62), fontsize=10.5, color=TEAL,
                arrowprops=dict(arrowstyle='->', color=TEAL, lw=1.3))
    ax.set_title('Graph diffusion drives node features toward uniformity',
                 fontsize=13.5, color=NAVY, pad=10, weight='bold')
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], loc='center right',
              fontsize=10.5, frameon=True, framealpha=0.95)
    drop = 100 * (dr[2] - dr[0]) / abs(dr[0])
    ax.text(0.985, 0.03,
            f'Dirichlet energy drops {abs(drop):.0f}% by K=2; cosine rising. '
            f'GCN op (normalized adjacency)^K, functional adjacency, n={n} obs.',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.2, color=GREY)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')


if __name__ == '__main__':
    main()
