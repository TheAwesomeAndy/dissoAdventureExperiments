#!/usr/bin/env python3
"""
run_chapter6_exp6_temporal.py — ARSPI-Net Chapter 6, Experiment 6.6
Sliding-Window Temporal Localization of Affective Detectability

Scientific Cycle: Motivation → Design → Raw Observation → Analysis → Conclusion

MATHEMATICAL MOTIVATION:
  Experiments 6.1-6.5 computed metrics over the full 1229-step epoch. But
  EEG is temporally structured — the ERP unfolds over time (P1~100ms,
  N2~200ms, P3~400ms, LPP~600-800ms). The reservoir has finite fading
  memory (tau_AC ~ 47 steps from Exp 6.3). Where in time does the
  reservoir carry affective discriminative information?

EXPERIMENTAL DESIGN:
  150-step sliding windows (146ms at 1024 Hz), 50-step stride (49ms),
  producing 22 windows across the 1229-step epoch. For each window:
  compute 7 validated metrics, then within-valence and cross-valence
  effect sizes.

Generates 4 figures:
  raw6_6a_time_resolved.pdf          — 3 subjects: pop rate + window metrics over time
  raw6_6b_grand_trajectories.pdf     — Grand-average metric trajectories (N=211)
  raw6_6c_subject_variability.pdf    — Spaghetti plots: individual subject variability
  analysis6_6d_temporal_detectability.pdf — Effect size curves + heatmaps + peak analysis

Results:
  Peak Cute-Erotic d = -0.83 at 708ms (vs d=-0.40 full-epoch: 2x amplification)
  Two temporal regimes: 400-500ms (efficiency/complexity) and 650-750ms (spike-counting)
  Maps onto P3/LPP onset and sustained LPP windows of ERP component timing.

Usage:
    python run_chapter6_exp6_temporal.py \
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

N_RES=256; BETA=0.05; M_TH=0.5; EXCLUDED={127}
PAT=re.compile(r'SHAPE_Community_(\d+)_IAPS(Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC\.txt')
CATS=['Threat','Mutilation','Cute','Erotic']
CCOL={'Threat':'#e74c3c','Mutilation':'#c0392b','Cute':'#27ae60','Erotic':'#2ecc71'}
WINDOW_METRICS=['total_spikes','mean_firing_rate','phi_proxy','temporal_sparsity',
                'permutation_entropy','rate_entropy','rate_variance']
SHORT_W=['Spikes','Rate','Phi','TempSp','H_pi','H_rate','Var_r']

class Res:
    def __init__(self,seed=42):
        rng=np.random.RandomState(seed);self.Win=rng.randn(N_RES)*0.3
        mask=(rng.rand(N_RES,N_RES)<0.1).astype(float);np.fill_diagonal(mask,0)
        self.Wrec=rng.randn(N_RES,N_RES)*0.05*mask;self.fanout=mask.sum(1)
    def run(self,u):
        T=len(u);M=np.zeros((T,N_RES));S=np.zeros((T,N_RES),dtype=np.int8)
        m=np.zeros(N_RES);s=np.zeros(N_RES)
        for t in range(T):
            I=self.Win*u[t]+self.Wrec@s;m=(1-BETA)*m*(1-s)+I
            sp=(m>=M_TH).astype(float);M[t]=m;S[t]=sp.astype(np.int8);s=sp
        return M,S

def compute_window_metrics(S_win,fanout):
    T,nr=S_win.shape
    if T<10:return {}
    pnr=S_win.mean(0);pr=S_win.mean(1);ts=float(S_win.sum())
    mets={'total_spikes':ts,'mean_firing_rate':float(S_win.mean())}
    synops=ts*(fanout.mean()+3)
    re_val=-np.sum(pnr*np.log2(pnr+1e-15)+(1-pnr)*np.log2(1-pnr+1e-15))/nr
    mets['rate_entropy']=float(re_val);mets['phi_proxy']=float(re_val/(synops+1e-10)*1e6)
    mets['temporal_sparsity']=float(np.mean(pr<1.0/nr))
    mets['rate_variance']=float(np.var(pnr))
    d=3;pats=defaultdict(int);nc=0
    for t in range(T-d+1):pats[tuple(np.argsort(pr[t:t+d]).tolist())]+=1;nc+=1
    if nc>0:
        p=np.array(list(pats.values()))/nc
        mets['permutation_entropy']=float(-np.sum(p*np.log2(p+1e-15))/np.log2(math.factorial(d)))
    else:mets['permutation_entropy']=0.0
    return mets

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

def norm(u):return(u-u.mean())/(u.std()+1e-10)

def main():
    p=argparse.ArgumentParser(description='Exp 6.6: Sliding-Window Temporal Localization')
    p.add_argument('--category-dirs',nargs='+',required=True)
    p.add_argument('--output-dir',default='figures');p.add_argument('--results-dir',default='results')
    p.add_argument('--channel',type=int,default=16)
    p.add_argument('--win-size',type=int,default=150,help='Window size in steps')
    p.add_argument('--stride',type=int,default=50,help='Stride in steps')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();os.makedirs(a.output_dir,exist_ok=True);os.makedirs(a.results_dir,exist_ok=True)
    t0=time.time();af=load_files(a.category_dirs);subjs=sorted(set(s for s,c in af.keys()))
    reservoir=Res(seed=a.seed)
    T_TOTAL=1229;WIN=a.win_size;STRIDE=a.stride
    wstarts=list(range(0,T_TOTAL-WIN+1,STRIDE))
    wcenters=[ws+WIN//2 for ws in wstarts]
    wcenters_ms=[c/1.024 for c in wcenters]
    nw=len(wstarts)
    print(f"Loaded {len(af)} files, {len(subjs)} subjects")
    print(f"Window: {WIN} steps ({WIN/1.024:.0f}ms), stride: {STRIDE} ({STRIDE/1.024:.0f}ms), {nw} windows")

    # ── FIG 1: Raw Observation — Time-resolved reservoir response ──
    dsids=[subjs[i] for i in [20,100,180]]
    fig=plt.figure(figsize=(22,20));gs=GridSpec(3,3,figure=fig,hspace=0.45,wspace=0.35)
    for si,sid in enumerate(dsids):
        # Col 1: pop rate
        ax=fig.add_subplot(gs[si,0])
        for cat in CATS:
            eeg=np.loadtxt(af[(sid,cat)]);u=norm(eeg[:,a.channel]);_,S=reservoir.run(u)
            ax.plot(np.arange(len(S))/1.024,uniform_filter1d(S.mean(1).astype(float),30),color=CCOL[cat],lw=0.8,alpha=0.8,label=cat if si==0 else '')
        ax.set_ylabel(f'S{sid}\nPop Rate',fontsize=10,fontweight='bold');ax.grid(alpha=0.1)
        if si==0:ax.legend(fontsize=7,ncol=2);ax.set_title('Population Rate',fontsize=11,fontweight='bold')
        for t_ms,lab in [(100,'P1'),(200,'N2'),(400,'P3'),(600,'LPP')]:
            ax.axvline(t_ms,color='gray',lw=0.5,ls=':')
            if si==0:ax.text(t_ms,ax.get_ylim()[1]*0.95,lab,fontsize=6,ha='center',color='gray')
        # Col 2: window rate
        ax=fig.add_subplot(gs[si,1])
        for cat in CATS:
            eeg=np.loadtxt(af[(sid,cat)]);u=norm(eeg[:,a.channel]);_,S=reservoir.run(u)
            wr=[S[ws:ws+WIN].mean() for ws in wstarts]
            ax.plot(wcenters_ms,wr,'o-',color=CCOL[cat],lw=1.0,ms=3)
        if si==0:ax.set_title('Window Rate',fontsize=11,fontweight='bold')
        ax.grid(alpha=0.2)
        # Col 3: window H_pi
        ax=fig.add_subplot(gs[si,2])
        for cat in CATS:
            eeg=np.loadtxt(af[(sid,cat)]);u=norm(eeg[:,a.channel]);_,S=reservoir.run(u)
            wh=[compute_window_metrics(S[ws:ws+WIN],reservoir.fanout).get('permutation_entropy',0) for ws in wstarts]
            ax.plot(wcenters_ms,wh,'o-',color=CCOL[cat],lw=1.0,ms=3)
        if si==0:ax.set_title('Window H_pi',fontsize=11,fontweight='bold')
        ax.grid(alpha=0.2)
    fig.suptitle(f'Raw Observation 6.17: Time-Resolved Reservoir (3 Subjects, Ch{a.channel})',fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/raw6_6a_time_resolved.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_6a ({time.time()-t0:.0f}s)")

    # ── Compute sliding-window metrics ──
    print("  Computing sliding-window metrics...")
    sw={}
    for si,sid in enumerate(subjs):
        if si%50==0:print(f"    {si+1}/{len(subjs)}... {time.time()-t0:.0f}s")
        for cat in CATS:
            if (sid,cat) not in af:continue
            eeg=np.loadtxt(af[(sid,cat)]);u=norm(eeg[:,a.channel]);_,S=reservoir.run(u)
            for wi,ws in enumerate(wstarts):
                sw[(sid,cat,wi)]=compute_window_metrics(S[ws:ws+WIN],reservoir.fanout)
    print(f"    {len(sw)} window observations ({time.time()-t0:.0f}s)")

    # ── FIG 2: Grand-average trajectories ──
    fig,axes=plt.subplots(2,4,figsize=(22,10))
    for mi,(met,lab) in enumerate(zip(WINDOW_METRICS,SHORT_W)):
        ax=axes[mi//4,mi%4]
        for cat in CATS:
            traj=[np.mean([sw[(sid,cat,wi)][met] for sid in subjs if (sid,cat,wi) in sw and met in sw[(sid,cat,wi)]]) for wi in range(nw)]
            ax.plot(wcenters_ms,traj,'o-',color=CCOL[cat],lw=1.5,ms=4,label=cat if mi==0 else '')
        ax.set_title(lab,fontsize=10,fontweight='bold');ax.grid(alpha=0.2)
        if mi==0:ax.legend(fontsize=7,ncol=2)
        for t_ms in [100,200,400,600]:ax.axvline(t_ms,color='gray',lw=0.4,ls=':')
    if len(WINDOW_METRICS)<8:axes[1,3].axis('off')
    fig.suptitle('Raw Observation 6.18: Grand-Average Metric Trajectories (N=211)',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_6b_grand_trajectories.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_6b ({time.time()-t0:.0f}s)")

    # ── FIG 3: Subject-level spaghetti ──
    fig,axes=plt.subplots(2,2,figsize=(18,12))
    sample=subjs[::7][:30]
    for ci,(cat,col_i) in enumerate([('Threat',0),('Cute',1)]):
        ax=axes[0,col_i]
        for sid in sample:
            if (sid,cat,0) not in sw:continue
            traj=[sw[(sid,cat,wi)]['mean_firing_rate'] for wi in range(nw)]
            ax.plot(wcenters_ms,traj,color=CCOL[cat],lw=0.3,alpha=0.3)
        grand=[np.mean([sw[(sid,cat,wi)]['mean_firing_rate'] for sid in subjs if (sid,cat,wi) in sw]) for wi in range(nw)]
        ax.plot(wcenters_ms,grand,'k-',lw=2.5,label='Grand avg')
        ax.set_title(f'{cat}: Rate',fontsize=11,fontweight='bold',color=CCOL[cat]);ax.legend(fontsize=8);ax.grid(alpha=0.1)
        ax=axes[1,col_i]
        for sid in sample:
            if (sid,cat,0) not in sw:continue
            traj=[sw[(sid,cat,wi)]['permutation_entropy'] for wi in range(nw)]
            ax.plot(wcenters_ms,traj,color=CCOL[cat],lw=0.3,alpha=0.3)
        grand=[np.mean([sw[(sid,cat,wi)]['permutation_entropy'] for sid in subjs if (sid,cat,wi) in sw]) for wi in range(nw)]
        ax.plot(wcenters_ms,grand,'k-',lw=2.5)
        ax.set_title(f'{cat}: H_pi',fontsize=11,fontweight='bold',color=CCOL[cat]);ax.set_xlabel('Window Center (ms)');ax.grid(alpha=0.1)
    fig.suptitle('Raw Observation 6.19: Subject-Level Variability (30 subjects)',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_6c_subject_variability.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_6c ({time.time()-t0:.0f}s)")

    # ── ANALYSIS: Time-resolved effect sizes ──
    print("  Computing time-resolved effect sizes...")
    ec={}
    for met in WINDOW_METRICS:
        ec[met]={'thr_mut_d':[],'cut_ero_d':[],'neg_pos_d':[],'thr_mut_p':[],'cut_ero_p':[]}
        for wi in range(nw):
            thr=np.array([sw[(s,'Threat',wi)][met] for s in subjs if (s,'Threat',wi) in sw and met in sw[(s,'Threat',wi)]])
            mut=np.array([sw[(s,'Mutilation',wi)][met] for s in subjs if (s,'Mutilation',wi) in sw and met in sw[(s,'Mutilation',wi)]])
            cut=np.array([sw[(s,'Cute',wi)][met] for s in subjs if (s,'Cute',wi) in sw and met in sw[(s,'Cute',wi)]])
            ero=np.array([sw[(s,'Erotic',wi)][met] for s in subjs if (s,'Erotic',wi) in sw and met in sw[(s,'Erotic',wi)]])
            dn=thr-mut;d_nm=dn.mean()/(dn.std()+1e-15)
            try:_,p_nm=wilcoxon(thr,mut)
            except:p_nm=1.0
            dp=cut-ero;d_ce=dp.mean()/(dp.std()+1e-15)
            try:_,p_ce=wilcoxon(cut,ero)
            except:p_ce=1.0
            nm=(thr+mut)/2;pm=(cut+ero)/2;dnp=nm-pm;d_np=dnp.mean()/(dnp.std()+1e-15)
            ec[met]['thr_mut_d'].append(d_nm);ec[met]['cut_ero_d'].append(d_ce);ec[met]['neg_pos_d'].append(d_np)
            ec[met]['thr_mut_p'].append(p_nm);ec[met]['cut_ero_p'].append(p_ce)

    # ── FIG 4: Analysis ──
    fig=plt.figure(figsize=(22,16));gs=GridSpec(3,4,figure=fig,hspace=0.5,wspace=0.35)
    # Row 1: effect size curves
    ax=fig.add_subplot(gs[0,0:2])
    for mi,(met,lab) in enumerate(zip(WINDOW_METRICS[:4],SHORT_W[:4])):
        ax.plot(wcenters_ms,ec[met]['cut_ero_d'],'-',lw=1.5,label=lab,alpha=0.8)
    ax.axhline(0,color='black',lw=1);ax.axhline(0.2,color='gray',ls=':',lw=0.8);ax.axhline(-0.2,color='gray',ls=':',lw=0.8)
    ax.set_xlabel('Window Center (ms)');ax.set_ylabel("d (Cute-Erotic)");ax.set_title('Within-Positive',fontsize=11,fontweight='bold')
    ax.legend(fontsize=7,ncol=2);ax.grid(alpha=0.2)
    for t in [100,200,400,600]:ax.axvline(t,color='gray',lw=0.4,ls=':')
    ax=fig.add_subplot(gs[0,2:4])
    for mi,(met,lab) in enumerate(zip(WINDOW_METRICS[:4],SHORT_W[:4])):
        ax.plot(wcenters_ms,ec[met]['thr_mut_d'],'-',lw=1.5,label=lab,alpha=0.8)
    ax.axhline(0,color='black',lw=1);ax.axhline(0.2,color='gray',ls=':',lw=0.8);ax.axhline(-0.2,color='gray',ls=':',lw=0.8)
    ax.set_xlabel('Window Center (ms)');ax.set_ylabel("d (Threat-Mutilation)");ax.set_title('Within-Negative',fontsize=11,fontweight='bold')
    ax.legend(fontsize=7,ncol=2);ax.grid(alpha=0.2)
    for t in [100,200,400,600]:ax.axvline(t,color='gray',lw=0.4,ls=':')
    # Row 2: heatmaps
    ax=fig.add_subplot(gs[1,0:2])
    dm=np.zeros((len(WINDOW_METRICS),nw))
    for mi,met in enumerate(WINDOW_METRICS):dm[mi,:]=ec[met]['cut_ero_d']
    im=ax.imshow(dm,aspect='auto',cmap='RdBu_r',vmin=-0.5,vmax=0.5,extent=[wcenters_ms[0],wcenters_ms[-1],len(WINDOW_METRICS)-0.5,-0.5])
    ax.set_yticks(range(len(WINDOW_METRICS)));ax.set_yticklabels(SHORT_W,fontsize=8);ax.set_xlabel('Window Center (ms)')
    plt.colorbar(im,ax=ax,shrink=0.6,label='d');ax.set_title('Cute-Erotic Heatmap',fontsize=11,fontweight='bold')
    ax=fig.add_subplot(gs[1,2:4])
    dm2=np.zeros((len(WINDOW_METRICS),nw))
    for mi,met in enumerate(WINDOW_METRICS):dm2[mi,:]=ec[met]['thr_mut_d']
    im=ax.imshow(dm2,aspect='auto',cmap='RdBu_r',vmin=-0.5,vmax=0.5,extent=[wcenters_ms[0],wcenters_ms[-1],len(WINDOW_METRICS)-0.5,-0.5])
    ax.set_yticks(range(len(WINDOW_METRICS)));ax.set_yticklabels(SHORT_W,fontsize=8);ax.set_xlabel('Window Center (ms)')
    plt.colorbar(im,ax=ax,shrink=0.6,label='d');ax.set_title('Threat-Mutilation Heatmap',fontsize=11,fontweight='bold')
    # Row 3: peak summary + significance
    ax=fig.add_subplot(gs[2,0:2])
    for mi,(met,lab) in enumerate(zip(WINDOW_METRICS,SHORT_W)):
        ds=np.abs(ec[met]['cut_ero_d']);pk=np.argmax(ds);pms=wcenters_ms[pk];pd=ec[met]['cut_ero_d'][pk]
        ax.barh(mi,pd,color='#27ae60' if abs(pd)>0.2 else '#95a5a6',edgecolor='black',lw=0.3)
        ax.text(pd+0.02*np.sign(pd),mi,f'{pms:.0f}ms (d={pd:.3f})',fontsize=7,va='center')
    ax.set_yticks(range(len(WINDOW_METRICS)));ax.set_yticklabels(SHORT_W,fontsize=8);ax.axvline(0,color='black',lw=1)
    ax.set_xlabel('Peak d (Cute-Erotic)');ax.set_title('Peak Window',fontsize=11,fontweight='bold')
    # Significance time course
    ax=fig.add_subplot(gs[2,2:4])
    bm=WINDOW_METRICS[np.argmax([max(np.abs(ec[m]['cut_ero_d'])) for m in WINDOW_METRICS])]
    bl=SHORT_W[WINDOW_METRICS.index(bm)];ps=ec[bm]['cut_ero_p']
    ax.plot(wcenters_ms,-np.log10(np.array(ps)+1e-15),'b-o',lw=1.5,ms=4,label=bl)
    ax.axhline(-np.log10(0.05),color='red',ls='--',lw=1.5,label='p=0.05')
    ax.axhline(-np.log10(0.001),color='orange',ls=':',lw=1.5,label='p=0.001')
    ax.set_xlabel('Window Center (ms)');ax.set_ylabel('-log10(p)');ax.set_title(f'Significance: {bl}',fontsize=11,fontweight='bold')
    ax.legend(fontsize=8);ax.grid(alpha=0.2)
    for t in [100,200,400,600]:ax.axvline(t,color='gray',lw=0.4,ls=':')
    pk=np.argmax(-np.log10(np.array(ps)+1e-15))
    ax.annotate(f'Peak: {wcenters_ms[pk]:.0f}ms\np={ps[pk]:.1e}',xy=(wcenters_ms[pk],-np.log10(ps[pk]+1e-15)),
                xytext=(wcenters_ms[pk]+100,-np.log10(ps[pk]+1e-15)*0.8),arrowprops=dict(arrowstyle='->'),fontsize=8,fontweight='bold')
    fig.suptitle(f'Analysis 6.6: Temporal Localization ({nw} windows, {WIN/1.024:.0f}ms, N={len(subjs)})',fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/analysis6_6d_temporal_detectability.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  analysis6_6d ({time.time()-t0:.0f}s)")

    pickle.dump({'effect_curves':ec,'window_metrics':WINDOW_METRICS,'window_centers_ms':wcenters_ms,
                 'win_size':WIN,'stride':STRIDE,'n_windows':nw,'n_subjects':len(subjs)},
                open(f'{a.results_dir}/ch6_exp6_full.pkl','wb'))

    print(f"\n  Key results:")
    for met,lab in zip(WINDOW_METRICS,SHORT_W):
        dce=ec[met]['cut_ero_d'];dtm=ec[met]['thr_mut_d']
        pce=np.argmax(np.abs(dce));ptm=np.argmax(np.abs(dtm))
        print(f"    {lab:<8} CE peak: {wcenters_ms[pce]:6.0f}ms d={dce[pce]:+.3f}  TM peak: {wcenters_ms[ptm]:6.0f}ms d={dtm[ptm]:+.3f}")
    print(f"\nDone. 4 figures. {time.time()-t0:.0f}s.")

if __name__=='__main__':main()
