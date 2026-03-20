#!/usr/bin/env python3
"""
run_chapter6_exp3_valueadd.py — ARSPI-Net Chapter 6, Experiment 6.3
Nonlinear Transformation Characterization: Four Processing Paths

Scientific Cycle: Motivation → Design → Raw Observation → Analysis → Conclusion

MATHEMATICAL MOTIVATION:
  The reservoir performs u(t) in R -> M(t) in R^256 via LIF dynamics.
  Cover's theorem predicts nonlinear projections into higher-dimensional
  spaces can make previously inseparable patterns separable. ENGINEERING
  QUESTION: Does the LIF expansion enhance temporal descriptor detectability,
  and is the enhancement SPECIFIC to the spike-threshold mechanism?

EXPERIMENTAL DESIGN:
  Four processing paths applied to the same 80 subjects x 4 categories:
    1. Raw EEG (no transformation — the baseline)
    2. LIF Reservoir (256 spiking neurons, the dissertation's front-end)
    3. Tanh-RNN (256 continuous neurons, same architecture, tanh activation)
    4. Random Fourier Features (256 cosine projections, memoryless)
  Three comparable temporal descriptors computed on all four paths:
    Permutation entropy, autocorrelation decay, LZ complexity.

Generates 5 figures (4 raw observation, 1 analysis):
  raw6_3_four_paths_observation.pdf   — The 4 transformations side by side
  raw6_3_four_path_distributions.pdf  — Population distributions x 4 paths
  raw6_3_four_path_differences.pdf    — Within-valence paired diffs x 4 paths
  raw6_3_four_path_profiles.pdf       — Individual subject H_pi profiles
  analysis6_3_four_path_comparison.pdf — Effect size comparison + heatmaps

Results:
  LIF H_pi Threat-Mutilation d=-0.49 (p=6e-5). Tanh-RNN: d=+0.44 (OPPOSITE SIGN).
  Raw EEG tau_AC d=-0.61 (strongest for amplitude). LIF tau_AC Cute-Erotic d=+0.41
  (creates signal where raw EEG has none). Each front-end creates a different
  measurement space. The LIF spike mechanism is non-substitutable.

Usage:
    python run_chapter6_exp3_valueadd.py \
        --category-dirs categoriesbatch1 ... --output-dir figures/ --results-dir results/

Author: Andrew (ARSPI-Net Dissertation), March 2026
"""
import numpy as np, os, re, pickle, time, math, argparse
from collections import defaultdict
from scipy.stats import wilcoxon
from scipy.ndimage import uniform_filter1d
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

N_RES=256;BETA=0.05;M_TH=0.5;EXCLUDED={127}
PAT=re.compile(r'SHAPE_Community_(\d+)_IAPS(Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC\.txt')
CATS=['Threat','Mutilation','Cute','Erotic']
CCOL={'Threat':'#e74c3c','Mutilation':'#c0392b','Cute':'#27ae60','Erotic':'#2ecc71'}
PATHS=['Raw EEG','LIF Reservoir','Tanh-RNN','Random Fourier']
PCOL={'Raw EEG':'#95a5a6','LIF Reservoir':'#2980b9','Tanh-RNN':'#e67e22','Random Fourier':'#8e44ad'}
METRICS=['permutation_entropy','tau_ac','lz_complexity']
MET_LABELS=['H_pi','tau_AC','LZ']

def norm(u):return(u-u.mean())/(u.std()+1e-10)
def perm_entropy(x,d=4):
    T=len(x);pats=defaultdict(int);nc=0
    for t in range(T-d+1):pats[tuple(np.argsort(x[t:t+d]).tolist())]+=1;nc+=1
    if nc>0:p_=np.array(list(pats.values()))/nc;return float(-np.sum(p_*np.log2(p_+1e-15))/np.log2(math.factorial(d)))
    return 0.0
def ac_decay(x):
    T=len(x);xc=x-x.mean();v=np.sum(xc**2)
    if v<=0:return 0.0
    ml=min(200,T//2);ac=np.array([np.sum(xc[:T-k]*xc[k:])/v for k in range(ml)])
    idx=np.where(ac<=np.exp(-1))[0];return float(idx[0]) if len(idx)>0 else float(ml)
def lz_bin(x_bin):
    n=len(x_bin);vocab=set();c=0;w=''
    for x in x_bin:
        wc=w+str(int(x))
        if wc not in vocab:vocab.add(wc);c+=1;w=''
        else:w=wc
    if w:c+=1
    return float(c/(n/(np.log2(n)+1e-10)))

class LIFRes:
    def __init__(self,seed=42):
        rng=np.random.RandomState(seed);self.Win=rng.randn(N_RES)*0.3
        mask=(rng.rand(N_RES,N_RES)<0.1).astype(float);np.fill_diagonal(mask,0)
        self.Wrec=rng.randn(N_RES,N_RES)*0.05*mask
    def run(self,u):
        T=len(u);M=np.zeros((T,N_RES));m=np.zeros(N_RES);s=np.zeros(N_RES)
        for t in range(T):
            I=self.Win*u[t]+self.Wrec@s;m=(1-BETA)*m*(1-s)+I
            sp=(m>=M_TH).astype(float);M[t]=m;s=sp
        return M.mean(1)
class TanhRNN:
    def __init__(self,seed=42):
        rng=np.random.RandomState(seed);self.Win=rng.randn(N_RES)*0.3
        mask=(rng.rand(N_RES,N_RES)<0.1).astype(float);np.fill_diagonal(mask,0)
        self.Wrec=rng.randn(N_RES,N_RES)*0.05*mask
    def run(self,u):
        T=len(u);H=np.zeros((T,N_RES));h=np.zeros(N_RES)
        for t in range(T):h=np.tanh((1-BETA)*h+self.Win*u[t]+self.Wrec@h);H[t]=h
        return H.mean(1)
class RandFourier:
    def __init__(self,seed=42):
        rng=np.random.RandomState(seed);self.W=rng.randn(N_RES)*3.0;self.b=rng.uniform(0,2*np.pi,N_RES)
    def run(self,u):
        T=len(u);H=np.zeros((T,N_RES))
        for t in range(T):H[t]=np.cos(self.W*u[t]+self.b)
        return H.mean(1)

def load_files(dirs):
    af={}
    for d in dirs:
        if not os.path.isdir(d):continue
        for f in os.listdir(d):
            m=PAT.match(f)
            if m:
                sid=int(m.group(1));cat=m.group(3)
                if sid not in EXCLUDED:af[(sid,cat)]=os.path.join(d,f)
    return af

def compute_met(traj, met):
    if met=='permutation_entropy': return perm_entropy(traj)
    elif met=='tau_ac': return ac_decay(traj)
    elif met=='lz_complexity': return lz_bin((traj>np.median(traj)).astype(int))
    return 0.0

def main():
    p=argparse.ArgumentParser(description='Exp 6.3: Four-Path Transformation Characterization')
    p.add_argument('--category-dirs',nargs='+',required=True)
    p.add_argument('--output-dir',default='figures');p.add_argument('--results-dir',default='results')
    p.add_argument('--n-subjects',type=int,default=80);p.add_argument('--channel',type=int,default=16)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();os.makedirs(a.output_dir,exist_ok=True);os.makedirs(a.results_dir,exist_ok=True)
    t0=time.time();af=load_files(a.category_dirs);subjs=sorted(set(s for s,c in af.keys()))
    lif=LIFRes(a.seed);tanh=TanhRNN(a.seed);rff=RandFourier(a.seed)
    front_ends={'Raw EEG':None,'LIF Reservoir':lif,'Tanh-RNN':tanh,'Random Fourier':rff}
    print(f"Loaded {len(af)} files, {len(subjs)} subjects")

    # ── FIG 1: The four transformations ──
    dsid=subjs[len(subjs)//4]
    eeg=np.loadtxt(af[(dsid,'Threat')]);u_raw=norm(eeg[:,a.channel])
    trajs={'Raw EEG':u_raw,'LIF Reservoir':lif.run(u_raw),'Tanh-RNN':tanh.run(u_raw),'Random Fourier':rff.run(u_raw)}
    fig=plt.figure(figsize=(20,20));gs=GridSpec(5,2,figure=fig,hspace=0.5,wspace=0.3)
    ax=fig.add_subplot(gs[0,:]);t_ms=np.arange(len(u_raw))/1.024
    ax.plot(t_ms,u_raw,'k-',lw=0.5);ax.set_ylabel('Amplitude');ax.set_title(f'Input: Raw EEG (S{dsid}, Ch{a.channel}, Threat)',fontsize=12,fontweight='bold');ax.grid(alpha=0.1)
    for ti,(lab,tr) in enumerate(trajs.items()):
        ax=fig.add_subplot(gs[1+ti//2,ti%2]);ax.plot(t_ms,tr,color=PCOL[lab],lw=0.6)
        hpi=perm_entropy(tr);tac=ac_decay(tr)
        ax.set_title(lab,fontsize=11,fontweight='bold',color=PCOL[lab]);ax.grid(alpha=0.1)
        ax.text(0.98,0.95,f'H_pi={hpi:.4f}\ntau_AC={tac:.0f}',transform=ax.transAxes,fontsize=9,ha='right',va='top',fontweight='bold',bbox=dict(boxstyle='round',facecolor='white',alpha=0.9))
    ax=fig.add_subplot(gs[3,:]);seg=slice(300,600)
    for lab,tr in trajs.items():ax.plot(np.arange(300)/1.024+300/1.024,tr[seg],color=PCOL[lab],lw=1.2,alpha=0.7,label=lab)
    ax.legend(fontsize=9,ncol=4);ax.set_title('Zoomed Overlay (300-600 steps)',fontsize=11,fontweight='bold');ax.grid(alpha=0.2)
    ax=fig.add_subplot(gs[4,0])
    for lab,tr in trajs.items():
        xc=tr-tr.mean();v=np.sum(xc**2)
        if v>0:ax.plot([np.sum(xc[:len(tr)-k]*xc[k:])/v for k in range(100)],color=PCOL[lab],lw=1.5,label=lab)
    ax.axhline(np.exp(-1),color='gray',ls=':');ax.set_title('Autocorrelation',fontsize=10,fontweight='bold');ax.legend(fontsize=7);ax.grid(alpha=0.2)
    ax=fig.add_subplot(gs[4,1]);x_=np.arange(4);w_=0.2
    for pi,(lab,fe) in enumerate(front_ends.items()):
        hv=[]
        for cat in CATS:
            ec=np.loadtxt(af[(dsid,cat)]);uc=norm(ec[:,a.channel])
            hv.append(perm_entropy(uc if fe is None else fe.run(uc)))
        ax.bar(x_+pi*w_,hv,w_,color=PCOL[lab],edgecolor='black',lw=0.3,alpha=0.8,label=lab)
    ax.set_xticks(x_+1.5*w_);ax.set_xticklabels(CATS,fontsize=9);ax.legend(fontsize=6);ax.set_title(f'H_pi x 4 Cats x 4 Paths (S{dsid})',fontsize=10,fontweight='bold')
    fig.suptitle('Raw Observation: Same Input, Four Nonlinear Processing Paths',fontsize=13,fontweight='bold',y=1.02)
    plt.savefig(f'{a.output_dir}/raw6_3_four_paths_observation.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  fig1 ({time.time()-t0:.0f}s)")

    # ── COMPUTE ──
    rng=np.random.RandomState(a.seed);tsubs=sorted(rng.choice(subjs,min(a.n_subjects,len(subjs)),replace=False))
    fp={pa:{m:{c:[] for c in CATS} for m in METRICS} for pa in PATHS}
    print(f"  Computing {len(tsubs)} subjects x 4 cats x 4 paths...")
    for si,sid in enumerate(tsubs):
        if si%20==0:print(f"    {si+1}/{len(tsubs)}... {time.time()-t0:.0f}s")
        for cat in CATS:
            if (sid,cat) not in af:continue
            eeg_=np.loadtxt(af[(sid,cat)]);u=norm(eeg_[:,a.channel])
            for pa,fe in front_ends.items():
                tr=u if fe is None else fe.run(u)
                for m in METRICS:fp[pa][m][cat].append(compute_met(tr,m))

    # ── FIG 2: Population distributions ──
    fig,axes=plt.subplots(3,4,figsize=(22,14))
    for mi,m in enumerate(METRICS):
        for pi,pa in enumerate(PATHS):
            ax=axes[mi,pi]
            for cat in CATS:ax.hist(fp[pa][m][cat],bins=20,alpha=0.4,color=CCOL[cat],edgecolor='black',lw=0.2,density=True);ax.axvline(np.mean(fp[pa][m][cat]),color=CCOL[cat],lw=1.5,ls='--')
            if pi==0:ax.set_ylabel(MET_LABELS[mi],fontsize=9,fontweight='bold')
            if mi==0:ax.set_title(pa,fontsize=11,fontweight='bold',color=PCOL[pa])
            ax.grid(alpha=0.1)
    fig.suptitle(f'Raw Observation: Distributions x 4 Paths (N={len(tsubs)})',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_3_four_path_distributions.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  fig2 ({time.time()-t0:.0f}s)")

    # ── FIG 3: Paired differences ──
    contrasts=[('Threat','Mutilation','Thr-Mut'),('Cute','Erotic','Cut-Ero')]
    fig,axes=plt.subplots(3,2,figsize=(18,14))
    for mi,m in enumerate(METRICS):
        for ci,(c1,c2,lab) in enumerate(contrasts):
            ax=axes[mi,ci]
            for pa in PATHS:
                d=np.array(fp[pa][m][c1])-np.array(fp[pa][m][c2])
                ax.hist(d,bins=25,alpha=0.35,color=PCOL[pa],edgecolor='black',lw=0.2,density=True,label=f'{pa} (u={d.mean():.4f})')
                ax.axvline(d.mean(),color=PCOL[pa],lw=2,ls='--')
            ax.axvline(0,color='black',lw=1.5)
            if ci==0:ax.set_ylabel(MET_LABELS[mi],fontsize=9,fontweight='bold')
            if mi==0:ax.set_title(lab,fontsize=12,fontweight='bold')
            ax.legend(fontsize=5);ax.grid(alpha=0.1)
    fig.suptitle('Raw Observation: Within-Valence Paired Differences x 4 Paths\nDifferent paths shift in DIFFERENT DIRECTIONS for the same contrast.',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_3_four_path_differences.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  fig3 ({time.time()-t0:.0f}s)")

    # ── FIG 4: Individual profiles ──
    fig,axes=plt.subplots(2,4,figsize=(22,10))
    for si in range(min(8,len(tsubs))):
        sid=tsubs[si];ax=axes[si//4,si%4];x_=np.arange(4);w_=0.2
        for pi,pa in enumerate(PATHS):
            hv=[fp[pa]['permutation_entropy'][cat][si] if si<len(fp[pa]['permutation_entropy'][cat]) else 0 for cat in CATS]
            ax.bar(x_+pi*w_,hv,w_,color=PCOL[pa],edgecolor='black',lw=0.2,alpha=0.7,label=pa if si==0 else '')
        ax.set_xticks(x_+1.5*w_);ax.set_xticklabels(['Thr','Mut','Cut','Ero'],fontsize=7)
        ax.set_title(f'S{sid}',fontsize=10,fontweight='bold');ax.grid(axis='y',alpha=0.1)
        if si==0:ax.legend(fontsize=5,ncol=2)
    fig.suptitle('Raw Observation: Individual H_pi Profiles x 4 Paths',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_3_four_path_profiles.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  fig4 ({time.time()-t0:.0f}s)")

    # ── ANALYSIS ──
    print("\n  Analysis:")
    analysis={}
    for pa in PATHS:
        analysis[pa]={}
        for m in METRICS:
            thr=np.array(fp[pa][m]['Threat']);mut=np.array(fp[pa][m]['Mutilation'])
            cut=np.array(fp[pa][m]['Cute']);ero=np.array(fp[pa][m]['Erotic'])
            dn=thr-mut;d_n=dn.mean()/(dn.std()+1e-15);dp=cut-ero;d_p=dp.mean()/(dp.std()+1e-15)
            try:_,pn=wilcoxon(thr,mut)
            except:pn=1.0
            try:_,pp=wilcoxon(cut,ero)
            except:pp=1.0
            analysis[pa][m]={'d_neg':float(d_n),'p_neg':float(pn),'d_pos':float(d_p),'p_pos':float(pp)}
            print(f"    {pa:<16} {m:<20} TM d={d_n:+.3f} p={pn:.1e}  CE d={d_p:+.3f} p={pp:.1e}")

    fig=plt.figure(figsize=(22,14));gs=GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.4)
    for ri,(contrast_key,title) in enumerate([('d_neg','Threat-Mutilation'),('d_pos','Cute-Erotic')]):
        ax=fig.add_subplot(gs[ri,0:2]);x_=np.arange(len(METRICS));w_=0.2
        for pi,pa in enumerate(PATHS):
            ds=[abs(analysis[pa][m][contrast_key]) for m in METRICS]
            ax.bar(x_+pi*w_,ds,w_,color=PCOL[pa],edgecolor='black',lw=0.3,label=pa,alpha=0.8)
        ax.set_xticks(x_+1.5*w_);ax.set_xticklabels(MET_LABELS,fontsize=9);ax.set_ylabel("|d|");ax.legend(fontsize=8)
        ax.set_title(f'{title}: |d| by Front-End',fontsize=11,fontweight='bold');ax.grid(axis='y',alpha=0.2)
        ax2=fig.add_subplot(gs[ri,2])
        dm=np.array([[analysis[pa][m][contrast_key] for m in METRICS] for pa in PATHS])
        im=ax2.imshow(dm,aspect='auto',cmap='RdBu_r',vmin=-0.7,vmax=0.7)
        ax2.set_xticks(range(len(METRICS)));ax2.set_xticklabels(MET_LABELS,fontsize=9)
        ax2.set_yticks(range(len(PATHS)));ax2.set_yticklabels(PATHS,fontsize=8)
        for pi_ in range(len(PATHS)):
            for mi_ in range(len(METRICS)):
                ax2.text(mi_,pi_,f'{dm[pi_,mi_]:+.2f}',ha='center',va='center',fontsize=8,fontweight='bold',color='white' if abs(dm[pi_,mi_])>0.35 else 'black')
        plt.colorbar(im,ax=ax2,shrink=0.6);ax2.set_title(f'{title} d (signed)',fontsize=10,fontweight='bold')
    fig.suptitle('Analysis: Four-Path Effect Size Comparison\nThe LIF spike mechanism produces effects that other nonlinearities do not replicate.',fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/analysis6_3_four_path_comparison.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  analysis ({time.time()-t0:.0f}s)")

    pickle.dump({'analysis':analysis,'paths':PATHS,'metrics':METRICS,'n_subjects':len(tsubs)},
                open(f'{a.results_dir}/ch6_exp3_unified.pkl','wb'))
    print(f"\nDone. 5 figures. {time.time()-t0:.0f}s.")

if __name__=='__main__':main()
