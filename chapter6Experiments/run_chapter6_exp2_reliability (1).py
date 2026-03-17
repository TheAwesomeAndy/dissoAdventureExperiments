#!/usr/bin/env python3
"""
run_chapter6_exp2_reliability.py — ARSPI-Net Chapter 6, Experiment 6.2
Cross-Initialization Reliability of Dynamical Metrics

Scientific Cycle: Motivation → Design → Raw Observation → Analysis → Conclusion

Generates 3 figures:
  raw6_2a_seed_comparison.pdf  — 3 seeds processing the same EEG (12 panels)
  raw6_2b_seed_scatter.pdf     — Seed-to-seed scatter for all 11 metrics
  analysis6_2c_reliability.pdf — ICC bar chart + gate decision table

Results: 9/11 pass ICC >= 0.75. 2 marginal. 0 fail.

Usage:
    python run_chapter6_exp2_reliability.py \
        --category-dirs categoriesbatch1 ... --output-dir figures/ --results-dir results/

Author: Andrew (ARSPI-Net Dissertation), March 2026
"""
import numpy as np, os, re, pickle, time, math, argparse
from collections import defaultdict
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d

N_RES=256;BETA=0.05;M_TH=0.5;EXCLUDED={127}
PAT=re.compile(r'SHAPE_Community_(\d+)_IAPS(Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC\.txt')
CATS=['Threat','Mutilation','Cute','Erotic']
METRICS=['total_spikes','mean_firing_rate','phi_proxy','population_sparsity',
         'temporal_sparsity','lz_complexity','permutation_entropy',
         'tau_relax','tau_ac','rate_entropy','rate_variance']
SHORT=['Spikes','Rate','Phi','PopSp','TempSp','LZ','H_pi','tau_r','tau_AC','H_rate','Var_r']

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

def compute_metrics(M,S,res):
    T,nr=S.shape;pnr=S.mean(0);pr=S.mean(1);ts=float(S.sum())
    mets={'total_spikes':ts,'mean_firing_rate':float(S.mean())}
    synops=ts*(res.fanout.mean()+3)
    re_val=-np.sum(pnr*np.log2(pnr+1e-15)+(1-pnr)*np.log2(1-pnr+1e-15))/nr
    mets['rate_entropy']=float(re_val);mets['phi_proxy']=float(re_val/(synops+1e-10)*1e6)
    mets['population_sparsity']=float((pnr.mean()**2)/(np.mean(pnr**2)+1e-15))
    mets['temporal_sparsity']=float(np.mean(pr<1.0/nr))
    mets['rate_variance']=float(np.var(pnr))
    neurons=np.random.RandomState(0).choice(nr,10,replace=False);lz=[]
    for ni in neurons:
        seq=S[:,ni];n=len(seq);vocab=set();c=0;w=''
        for x in seq:
            wc=w+str(int(x))
            if wc not in vocab:vocab.add(wc);c+=1;w=''
            else:w=wc
        if w:c+=1
        lz.append(c/(n/(np.log2(n)+1e-10)) if n>0 else 0)
    mets['lz_complexity']=float(np.mean(lz))
    Mm=M.mean(1);d=4;pats=defaultdict(int);nc=0
    for t in range(T-d+1):pats[tuple(np.argsort(Mm[t:t+d]).tolist())]+=1;nc+=1
    if nc>0:
        p=np.array(list(pats.values()))/nc
        mets['permutation_entropy']=float(-np.sum(p*np.log2(p+1e-15))/np.log2(math.factorial(d)))
    else:mets['permutation_entropy']=0.0
    ps=np.convolve(pr,np.ones(20)/20,mode='same');tp=np.argmax(ps[100:])+100
    if tp<T-50:
        dec=ps[tp:];rp=dec[0];ri=dec[-50:].mean()
        if rp>ri+0.001:tgt=ri+(rp-ri)/np.e;idx=np.where(dec<=tgt)[0];mets['tau_relax']=float(idx[0]) if len(idx)>0 else float(len(dec))
        else:mets['tau_relax']=0.0
    else:mets['tau_relax']=0.0
    pc=pr-pr.mean();vp=np.sum(pc**2)
    if vp>0:
        ml=min(200,T//2);ac=np.array([np.sum(pc[:T-k]*pc[k:])/vp for k in range(ml)])
        idx=np.where(ac<=np.exp(-1))[0];mets['tau_ac']=float(idx[0]) if len(idx)>0 else float(ml)
    else:mets['tau_ac']=0.0
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
    p=argparse.ArgumentParser(description='Exp 6.2: Reliability')
    p.add_argument('--category-dirs',nargs='+',required=True)
    p.add_argument('--output-dir',default='figures');p.add_argument('--results-dir',default='results')
    p.add_argument('--n-seeds',type=int,default=10);p.add_argument('--n-subjects',type=int,default=60)
    p.add_argument('--channels',nargs='+',type=int,default=[0,16,33])
    a=p.parse_args();os.makedirs(a.output_dir,exist_ok=True);os.makedirs(a.results_dir,exist_ok=True)
    t0=time.time();af=load_files(a.category_dirs);subjs=sorted(set(s for s,c in af.keys()))
    rng=np.random.RandomState(42);rsub=sorted(rng.choice(subjs,min(a.n_subjects,len(subjs)),replace=False))
    dsid=subjs[len(subjs)//4];dch=16
    print(f"Exp 6.2: {a.n_seeds} seeds x {len(rsub)} subjects x {len(CATS)} cats x {len(a.channels)} ch")

    # ── FIG 1: What different seeds look like ──
    eeg=np.loadtxt(af[(dsid,'Threat')]);u=norm(eeg[:,dch])
    res0=Res(42);res1=Res(1042);res2=Res(9042)
    M0,S0=res0.run(u);M1,S1=res1.run(u);M2,S2=res2.run(u)
    fig=plt.figure(figsize=(20,16));gs=GridSpec(4,3,figure=fig,hspace=0.5,wspace=0.35)
    for si,(S,lab,col) in enumerate([(S0,'Seed0','#3498db'),(S1,'Seed1','#e74c3c'),(S2,'Seed9','#27ae60')]):
        ax=fig.add_subplot(gs[0,si]);t_,n_=np.where(S[:,:50].T>0)
        ax.scatter(n_,t_,s=0.1,c=col,alpha=0.4,rasterized=True)
        ax.set_title(f'{lab}: {int(S.sum()):,} spikes',fontsize=10,fontweight='bold')
    for ni,n in enumerate([10,80,180]):
        ax=fig.add_subplot(gs[1,ni])
        ax.plot(M0[200:500,n],'b-',lw=0.6,alpha=0.7);ax.plot(M1[200:500,n],'r-',lw=0.6,alpha=0.7);ax.plot(M2[200:500,n],'g-',lw=0.6,alpha=0.7)
        ax.axhline(M_TH,color='gray',ls=':',lw=0.5);ax.set_title(f'Neuron {n}',fontsize=10,fontweight='bold')
    ax=fig.add_subplot(gs[2,0:2])
    ax.plot(uniform_filter1d(S0.mean(1).astype(float),20),'b-',lw=1,alpha=0.7,label='Seed0')
    ax.plot(uniform_filter1d(S1.mean(1).astype(float),20),'r-',lw=1,alpha=0.7,label='Seed1')
    ax.plot(uniform_filter1d(S2.mean(1).astype(float),20),'g-',lw=1,alpha=0.7,label='Seed9')
    ax.legend(fontsize=9);ax.set_title('Population Rate',fontsize=10,fontweight='bold')
    for si,(M,cm) in enumerate([(M0,'Blues'),(M1,'Reds'),(M2,'Greens')]):
        ax=fig.add_subplot(gs[3,si]);z=PCA(2,random_state=42).fit_transform(M)
        ax.scatter(z[:,0],z[:,1],c=np.arange(len(z)),cmap=cm,s=0.5,alpha=0.4,rasterized=True)
        ax.set_title(f'Seed {[0,1,9][si]} Phase',fontsize=10,fontweight='bold')
    fig.suptitle(f'Raw Observation 6.6: 3 Seeds, Same EEG (S{dsid}, Ch{dch})',fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/raw6_2a_seed_comparison.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_2a ({time.time()-t0:.0f}s)")

    # ── Compute metrics across seeds ──
    am={seed:{} for seed in range(a.n_seeds)}
    for seed in range(a.n_seeds):
        res=Res(seed=seed*1000+42);t1=time.time()
        for sid in rsub:
            for cat in CATS:
                if (sid,cat) not in af:continue
                eeg=np.loadtxt(af[(sid,cat)])
                for ch in a.channels:
                    u=norm(eeg[:,ch]);M,S=res.run(u);am[seed][(sid,cat,ch)]=compute_metrics(M,S,res)
        print(f"  Seed {seed+1}/{a.n_seeds}: {time.time()-t1:.0f}s")
    ok=list(am[0].keys());no=len(ok)

    # ── FIG 2: Seed scatter ──
    fig,axes=plt.subplots(3,4,figsize=(20,14))
    for mi,met in enumerate(METRICS):
        if mi>=11:break
        ax=axes[mi//4,mi%4];v0=[am[0][k][met] for k in ok];v1=[am[1][k][met] for k in ok]
        ax.scatter(v0,v1,s=2,alpha=0.3,color='#3498db');mn=min(min(v0),min(v1));mx=max(max(v0),max(v1))
        ax.plot([mn,mx],[mn,mx],'k--',lw=0.8);r=np.corrcoef(v0,v1)[0,1]
        ax.set_title(f'{SHORT[mi]}: r={r:.4f}',fontsize=10,fontweight='bold',color='green' if r>0.9 else 'orange' if r>0.7 else 'red')
        ax.tick_params(labelsize=7)
    axes[2,3].axis('off')
    fig.suptitle('Raw Observation 6.7: Seed-to-Seed Metric Agreement',fontsize=12,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_2b_seed_scatter.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_2b ({time.time()-t0:.0f}s)")

    # ── ICC Analysis ──
    rel={}
    for met in METRICS:
        mat=np.zeros((no,a.n_seeds))
        for si,k in enumerate(ok):
            for seed in range(a.n_seeds):mat[si,seed]=am[seed].get(k,{}).get(met,0)
        v=mat.std(1)>1e-15;mv=mat[v];n,k_=mv.shape
        if n<10:rel[met]={'icc':0,'mean_corr':0,'mean_rank_corr':0,'cv_mean':999};continue
        gm=mv.mean();rm=mv.mean(1);cm=mv.mean(0)
        SS_r=k_*np.sum((rm-gm)**2);SS_t=np.sum((mv-gm)**2);SS_c=n*np.sum((cm-gm)**2);SS_e=SS_t-SS_r-SS_c
        MS_r=SS_r/max(n-1,1);MS_e=SS_e/max((n-1)*(k_-1),1)
        icc=(MS_r-MS_e)/(MS_r+(k_-1)*MS_e) if (MS_r+(k_-1)*MS_e)>0 else 0
        corrs=[];rcorrs=[]
        for s1 in range(a.n_seeds):
            for s2 in range(s1+1,a.n_seeds):
                v1=mv[:,s1];v2=mv[:,s2]
                if v1.std()>0 and v2.std()>0:corrs.append(np.corrcoef(v1,v2)[0,1]);rc,_=spearmanr(v1,v2);rcorrs.append(rc)
        cv=mv.std(1)/(np.abs(mv.mean(1))+1e-15)
        rel[met]={'icc':float(icc),'mean_corr':float(np.mean(corrs)) if corrs else 0,'mean_rank_corr':float(np.mean(rcorrs)) if rcorrs else 0,'cv_mean':float(np.mean(cv))}

    passed=[m for m in METRICS if rel[m]['icc']>=0.75]
    marginal=[m for m in METRICS if 0.5<=rel[m]['icc']<0.75]
    failed=[m for m in METRICS if rel[m]['icc']<0.5]

    # ── FIG 3: Analysis ──
    fig=plt.figure(figsize=(20,10));gs=GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.35)
    ax=fig.add_subplot(gs[0,0:2]);iccs=[rel[m]['icc'] for m in METRICS]
    cols=['#27ae60' if v>=0.75 else '#f39c12' if v>=0.5 else '#e74c3c' for v in iccs]
    ax.bar(range(len(METRICS)),iccs,color=cols,edgecolor='black',lw=0.5)
    ax.axhline(0.75,color='green',ls='--',lw=1.5,label='Good');ax.axhline(0.5,color='orange',ls='--',lw=1.5,label='Moderate')
    ax.set_xticks(range(len(METRICS)));ax.set_xticklabels(SHORT,rotation=45,ha='right',fontsize=8);ax.set_ylim(0,1.05)
    ax.set_title(f'ICC(3,1): {len(passed)} pass, {len(marginal)} marginal',fontsize=11,fontweight='bold');ax.legend(fontsize=8)
    for i,v in enumerate(iccs):ax.text(i,v+0.02,f'{v:.3f}',ha='center',fontsize=7,fontweight='bold')
    ax=fig.add_subplot(gs[0,2]);ax.axis('off')
    td=[[m.replace('_',' ')[:15],f"{rel[m]['icc']:.3f}",f"{rel[m]['mean_corr']:.3f}",f"{rel[m]['cv_mean']:.3f}",'PASS' if rel[m]['icc']>=0.75 else 'MARG' if rel[m]['icc']>=0.5 else 'FAIL'] for m in METRICS]
    tab=ax.table(cellText=td,colLabels=['Metric','ICC','r','CV','Gate'],loc='center',cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(7);tab.scale(1,1.4)
    for i,m in enumerate(METRICS):
        c='#d5f5e3' if rel[m]['icc']>=0.75 else '#fdebd0' if rel[m]['icc']>=0.5 else '#fadbd8'
        for j in range(5):tab[i+1,j].set_facecolor(c)
    ax.set_title('Gate Decision',fontsize=11,fontweight='bold',pad=15)
    ax=fig.add_subplot(gs[1,0:2]);cvs=[rel[m]['cv_mean'] for m in METRICS]
    ax.bar(range(len(METRICS)),cvs,color=cols,edgecolor='black',lw=0.5)
    ax.set_xticks(range(len(METRICS)));ax.set_xticklabels(SHORT,rotation=45,ha='right',fontsize=8)
    ax.set_ylabel('CV');ax.set_title('Coefficient of Variation',fontsize=11,fontweight='bold')
    ax=fig.add_subplot(gs[1,2]);sm=sorted(METRICS,key=lambda m:-rel[m]['icc'])
    for i,m in enumerate(sm):
        c='#27ae60' if rel[m]['icc']>=0.75 else '#f39c12' if rel[m]['icc']>=0.5 else '#e74c3c'
        ax.barh(i,rel[m]['icc'],color=c,edgecolor='black',lw=0.3);ax.text(rel[m]['icc']+0.01,i,f"{rel[m]['icc']:.3f}",va='center',fontsize=8)
    ax.set_yticks(range(len(sm)));ax.set_yticklabels([m[:15] for m in sm],fontsize=7)
    ax.axvline(0.75,color='green',ls='--');ax.set_title('Ranking',fontsize=11,fontweight='bold')
    fig.suptitle(f'Analysis 6.2: Reliability Gate — {len(passed)}/11 Pass',fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/analysis6_2c_reliability.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  analysis6_2c ({time.time()-t0:.0f}s)")

    pickle.dump({'reliability_results':rel,'all_metrics':am,'metric_names':METRICS,
                 'n_seeds':a.n_seeds,'passed':passed,'marginal':marginal,'failed':failed},
                open(f'{a.results_dir}/ch6_exp2_full.pkl','wb'))
    print(f"\nDone. 3 figures. {time.time()-t0:.0f}s. Passed: {passed}")

if __name__=='__main__':main()
