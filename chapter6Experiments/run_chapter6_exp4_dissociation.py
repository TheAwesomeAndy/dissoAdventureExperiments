#!/usr/bin/env python3
"""
run_chapter6_exp4_dissociation.py — ARSPI-Net Chapter 6, Experiment 6.4
Fine-Grained Affective Subcategory Dissociation

Scientific Cycle: Motivation → Design → Raw Observation → Analysis → Conclusion

MATHEMATICAL MOTIVATION:
  Experiments 6.1-6.3 established the reservoir as a stable, reliable,
  temporally sensitive measurement instrument. The validated metrics capture
  genuine phase-dependent temporal organization. The question: do these
  metrics dissociate fine-grained affective subcategories beyond broad valence?

  Threat (rapid defensive mobilization) vs Mutilation (sustained disgust/horror)
  Cute (affiliative warmth) vs Erotic (sexual arousal, high arousal + positive)

Generates 6 figures:
  raw6_4a_category_dynamics.pdf    — 4 subjects x 4 cats: pop rate + phase portraits
  raw6_4b_metric_distributions.pdf — Violin + scatter for all 9 metrics x 4 cats (N=211)
  raw6_4c_within_valence_diffs.pdf — Paired difference histograms (Thr-Mut, Cut-Ero)
  raw6_4d_channel_profiles.pdf     — Per-channel spatial metric maps by category
  raw6_4e_individual_profiles.pdf  — 6 individual subject metric profiles
  analysis6_4f_dissociation.pdf    — Effect size heatmap + key scatter + boxplots

Results:
  Within-Negative: Permutation entropy d=-0.31 (p=8.5e-6), Mutilation > Threat
  Within-Positive: Total spikes d=-0.40 (p=1.9e-7), Erotic > Cute
  Arousal is the dominant dimension. Phi_proxy reveals efficiency asymmetry:
  Threat produces more efficient coding than Mutilation (fast defense vs slow horror).

Usage:
    python run_chapter6_exp4_dissociation.py \
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
from sklearn.decomposition import PCA

N_RES=256; BETA=0.05; M_TH=0.5; EXCLUDED={127}
PAT=re.compile(r'SHAPE_Community_(\d+)_IAPS(Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC\.txt')
CATS=['Threat','Mutilation','Cute','Erotic']
CCOL={'Threat':'#e74c3c','Mutilation':'#c0392b','Cute':'#27ae60','Erotic':'#2ecc71'}
VALIDATED=['total_spikes','mean_firing_rate','phi_proxy','temporal_sparsity',
           'permutation_entropy','tau_relax','tau_ac','rate_entropy','rate_variance']

class Res:
    def __init__(self,seed=42):
        rng=np.random.RandomState(seed); self.Win=rng.randn(N_RES)*0.3
        mask=(rng.rand(N_RES,N_RES)<0.1).astype(float); np.fill_diagonal(mask,0)
        self.Wrec=rng.randn(N_RES,N_RES)*0.05*mask; self.fanout=mask.sum(1)
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
    p=argparse.ArgumentParser(description='Exp 6.4: Subcategory Dissociation')
    p.add_argument('--category-dirs',nargs='+',required=True)
    p.add_argument('--output-dir',default='figures');p.add_argument('--results-dir',default='results')
    p.add_argument('--channels',nargs='+',type=int,default=[0,8,16,24,33])
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();os.makedirs(a.output_dir,exist_ok=True);os.makedirs(a.results_dir,exist_ok=True)
    t0=time.time();af=load_files(a.category_dirs);subjs=sorted(set(s for s,c in af.keys()))
    reservoir=Res(seed=a.seed)
    print(f"Loaded {len(af)} files, {len(subjs)} subjects")

    # ── FIG 1: Raw Observation — Category Dynamics (4 subjects) ──
    demo_sids=[subjs[i] for i in [10,60,120,180]]; dch=16
    fig=plt.figure(figsize=(20,20));gs=GridSpec(4,4,figure=fig,hspace=0.5,wspace=0.35)
    for si,sid in enumerate(demo_sids):
        ax=fig.add_subplot(gs[si,0:2])
        for cat in CATS:
            eeg=np.loadtxt(af[(sid,cat)]);u=norm(eeg[:,dch]);M,S=reservoir.run(u)
            ax.plot(uniform_filter1d(S.mean(1).astype(float),20),color=CCOL[cat],lw=1.0,alpha=0.8,label=cat if si==0 else '')
        ax.set_ylabel(f'S{sid}\nPop Rate',fontsize=10,fontweight='bold');ax.grid(alpha=0.1)
        if si==0:ax.legend(fontsize=8,ncol=4);ax.set_title('Population Rate',fontsize=11,fontweight='bold')
        for ci,(cat,cmap) in enumerate([('Threat','Reds'),('Cute','Greens')]):
            ax=fig.add_subplot(gs[si,2+ci]);eeg=np.loadtxt(af[(sid,cat)]);u=norm(eeg[:,dch]);M,S=reservoir.run(u)
            z=PCA(2,random_state=42).fit_transform(M);ax.scatter(z[:,0],z[:,1],c=np.arange(len(z)),cmap=cmap,s=0.5,alpha=0.4,rasterized=True)
            if si==0:ax.set_title(f'{cat} Phase',fontsize=10,fontweight='bold',color=CCOL[cat])
    fig.suptitle('Raw Observation 6.9: Four Subjects x Four Categories',fontsize=13,fontweight='bold',y=1.02)
    plt.savefig(f'{a.output_dir}/raw6_4a_category_dynamics.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_4a ({time.time()-t0:.0f}s)")

    # ── Compute all metrics ──
    print("  Computing metrics (full dataset)...")
    all_mets={}
    for si,sid in enumerate(subjs):
        if si%50==0:print(f"    {si+1}/{len(subjs)}... {time.time()-t0:.0f}s")
        for cat in CATS:
            if (sid,cat) not in af:continue
            eeg=np.loadtxt(af[(sid,cat)])
            for ch in a.channels:
                u=norm(eeg[:,ch]);M,S=reservoir.run(u);all_mets[(sid,cat,ch)]=compute_metrics(M,S,reservoir)

    # Aggregate per-subject means
    sm={cat:{m:[] for m in VALIDATED} for cat in CATS}
    for sid in subjs:
        for cat in CATS:
            vals={m:[] for m in VALIDATED}
            for ch in a.channels:
                k=(sid,cat,ch)
                if k in all_mets:
                    for m in VALIDATED:vals[m].append(all_mets[k][m])
            for m in VALIDATED:
                if vals[m]:sm[cat][m].append(np.mean(vals[m]))

    # ── FIG 2: Raw Observation — Metric Distributions ──
    fig,axes=plt.subplots(3,3,figsize=(20,16))
    for mi,met in enumerate(VALIDATED):
        ax=axes[mi//3,mi%3];data=[sm[cat][met] for cat in CATS];pos=[1,2,3.5,4.5]
        vp=ax.violinplot(data,positions=pos,showmeans=True,showextrema=False)
        for i,(body,cat) in enumerate(zip(vp['bodies'],CATS)):body.set_facecolor(CCOL[cat]);body.set_alpha(0.4)
        rng_j=np.random.RandomState(42)
        for i,(cat,p_) in enumerate(zip(CATS,pos)):
            jit=rng_j.uniform(-0.15,0.15,len(data[i]));ax.scatter(p_+jit,data[i],s=2,alpha=0.2,color=CCOL[cat],rasterized=True)
        ax.set_xticks(pos);ax.set_xticklabels(CATS,fontsize=8,rotation=20)
        for i,(cat,p_) in enumerate(zip(CATS,pos)):ax.text(p_,np.mean(data[i]),f'{np.mean(data[i]):.3f}',fontsize=6,ha='center',va='bottom',fontweight='bold')
        ax.set_title(met.replace('_',' '),fontsize=10,fontweight='bold');ax.grid(axis='y',alpha=0.1);ax.axvline(2.75,color='gray',lw=0.5,ls=':')
    fig.suptitle('Raw Observation 6.10: Metric Distributions by Subcategory (N=211)',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_4b_metric_distributions.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_4b ({time.time()-t0:.0f}s)")

    # ── FIG 3: Raw Observation — Within-Valence Paired Diffs ──
    fig,axes=plt.subplots(3,3,figsize=(20,16))
    for mi,met in enumerate(VALIDATED):
        ax=axes[mi//3,mi%3]
        dn=np.array(sm['Threat'][met])-np.array(sm['Mutilation'][met])
        dp=np.array(sm['Cute'][met])-np.array(sm['Erotic'][met])
        ax.hist(dn,bins=30,alpha=0.5,color='#e74c3c',edgecolor='black',lw=0.3,label=f'Thr-Mut (d={dn.mean()/(dn.std()+1e-15):.3f})',density=True)
        ax.hist(dp,bins=30,alpha=0.5,color='#27ae60',edgecolor='black',lw=0.3,label=f'Cut-Ero (d={dp.mean()/(dp.std()+1e-15):.3f})',density=True)
        ax.axvline(0,color='black',lw=1.5);ax.axvline(dn.mean(),color='#e74c3c',lw=2,ls='--');ax.axvline(dp.mean(),color='#27ae60',lw=2,ls='--')
        ax.set_title(met.replace('_',' '),fontsize=10,fontweight='bold');ax.legend(fontsize=7);ax.grid(alpha=0.1)
    fig.suptitle('Raw Observation 6.11: Within-Valence Paired Differences (N=211)',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_4c_within_valence_diffs.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_4c ({time.time()-t0:.0f}s)")

    # ── FIG 4: Raw Observation — Per-Channel Profiles ──
    fig,axes=plt.subplots(3,3,figsize=(20,14))
    for mi,met in enumerate(VALIDATED):
        ax=axes[mi//3,mi%3]
        for cat in CATS:
            ch_means=[np.mean([all_mets[(sid,cat,ch)][met] for sid in subjs if (sid,cat,ch) in all_mets]) for ch in a.channels]
            ax.plot(a.channels,ch_means,'o-',color=CCOL[cat],lw=1.5,ms=6,label=cat if mi==0 else '')
        ax.set_xlabel('Channel');ax.set_title(met.replace('_',' '),fontsize=10,fontweight='bold');ax.grid(alpha=0.2)
        if mi==0:ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Raw Observation 6.12: Per-Channel Metric Profiles by Category',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_4d_channel_profiles.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_4d ({time.time()-t0:.0f}s)")

    # ── FIG 5: Raw Observation — Individual Subject Profiles ──
    fig,axes=plt.subplots(2,3,figsize=(18,12))
    psids=[subjs[i] for i in [5,40,80,120,160,200]];pmets=['mean_firing_rate','permutation_entropy','tau_ac','rate_entropy']
    for si,sid in enumerate(psids):
        ax=axes[si//3,si%3];x=np.arange(len(pmets));w=0.18
        for ci,cat in enumerate(CATS):
            vals=[np.mean([all_mets[(sid,cat,ch)][m] for ch in a.channels if (sid,cat,ch) in all_mets]) for m in pmets]
            ax.bar(x+ci*w,vals,w,color=CCOL[cat],edgecolor='black',lw=0.3,label=cat if si==0 else '')
        ax.set_xticks(x+1.5*w);ax.set_xticklabels([m[:10] for m in pmets],fontsize=7,rotation=20)
        ax.set_title(f'Subject {sid}',fontsize=10,fontweight='bold');ax.grid(axis='y',alpha=0.1)
        if si==0:ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Raw Observation 6.13: Individual Subject Profiles',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_4e_individual_profiles.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_4e ({time.time()-t0:.0f}s)")

    # ── ANALYSIS: Statistical Dissociation ──
    print("\n  Statistical dissociation:")
    ar={}
    for met in VALIDATED:
        thr=np.array(sm['Threat'][met]);mut=np.array(sm['Mutilation'][met])
        cut=np.array(sm['Cute'][met]);ero=np.array(sm['Erotic'][met])
        dn=thr-mut;d_n=dn.mean()/(dn.std()+1e-15)
        try:_,p_n=wilcoxon(thr,mut)
        except:p_n=1.0
        dp=cut-ero;d_p=dp.mean()/(dp.std()+1e-15)
        try:_,p_p=wilcoxon(cut,ero)
        except:p_p=1.0
        nm=(thr+mut)/2;pm_=(cut+ero)/2;dc=nm-pm_;d_c=dc.mean()/(dc.std()+1e-15)
        ar[met]={'d_neg':float(d_n),'p_neg':float(p_n),'d_pos':float(d_p),'p_pos':float(p_p),'d_cross':float(d_c),
                 'means':{c:float(np.mean(sm[c][met])) for c in CATS}}
        print(f"    {met:<22} Thr-Mut d={d_n:+.4f} p={p_n:.1e}  Cut-Ero d={d_p:+.4f} p={p_p:.1e}  Cross d={d_c:+.4f}")

    # ── FIG 6: Analysis Figure ──
    fig=plt.figure(figsize=(22,14));gs=GridSpec(3,4,figure=fig,hspace=0.55,wspace=0.4)
    # Heatmap
    ax=fig.add_subplot(gs[0,0:3]);contrasts=['Thr-Mut\n(within neg)','Cut-Ero\n(within pos)','Neg-Pos\n(cross val)']
    dm=np.zeros((len(VALIDATED),3))
    for mi,met in enumerate(VALIDATED):r=ar[met];dm[mi,0]=r['d_neg'];dm[mi,1]=r['d_pos'];dm[mi,2]=r['d_cross']
    im=ax.imshow(dm,aspect='auto',cmap='RdBu_r',vmin=-0.5,vmax=0.5)
    ax.set_xticks(range(3));ax.set_xticklabels(contrasts,fontsize=10)
    ax.set_yticks(range(len(VALIDATED)));ax.set_yticklabels([m.replace('_',' ')[:12] for m in VALIDATED],fontsize=8)
    plt.colorbar(im,ax=ax,shrink=0.8,label="Cohen's d")
    for mi,met in enumerate(VALIDATED):
        r=ar[met]
        for ci,(dv,pv) in enumerate([(r['d_neg'],r['p_neg']),(r['d_pos'],r['p_pos']),(r['d_cross'],1.0)]):
            ax.text(ci,mi,f'{dv:.3f}',ha='center',va='center',fontsize=7,fontweight='bold' if abs(dv)>0.1 else 'normal',color='white' if abs(dv)>0.3 else 'black')
            if pv<0.05:ax.text(ci+0.35,mi-0.3,'*',fontsize=10,fontweight='bold')
    ax.set_title("Effect Sizes (* = p<0.05)",fontsize=11,fontweight='bold')
    # Ranking
    ax=fig.add_subplot(gs[0,3]);mw=[max(abs(ar[m]['d_neg']),abs(ar[m]['d_pos'])) for m in VALIDATED]
    si_=np.argsort(mw)[::-1];cols=['#27ae60' if mw[i]>0.15 else '#f39c12' if mw[i]>0.05 else '#95a5a6' for i in si_]
    ax.barh(range(len(VALIDATED)),[mw[i] for i in si_],color=cols,edgecolor='black',lw=0.3)
    ax.set_yticks(range(len(VALIDATED)));ax.set_yticklabels([VALIDATED[i][:12] for i in si_],fontsize=7)
    ax.axvline(0.1,color='orange',ls='--',lw=1.5);ax.set_title('Within-Val Ranking',fontsize=10,fontweight='bold')
    # Key scatters
    bn=VALIDATED[np.argmax([abs(ar[m]['d_neg']) for m in VALIDATED])]
    ax=fig.add_subplot(gs[1,0:2]);tv=sm['Threat'][bn];mv=sm['Mutilation'][bn]
    ax.scatter(tv,mv,s=8,alpha=0.3,color='#e74c3c');mn_=min(min(tv),min(mv));mx_=max(max(tv),max(mv));ax.plot([mn_,mx_],[mn_,mx_],'k--')
    ax.set_xlabel('Threat');ax.set_ylabel('Mutilation');ax.set_title(f'Within-Neg: {bn}\nd={ar[bn]["d_neg"]:.4f}, p={ar[bn]["p_neg"]:.1e}',fontsize=10,fontweight='bold');ax.grid(alpha=0.2)
    bp=VALIDATED[np.argmax([abs(ar[m]['d_pos']) for m in VALIDATED])]
    ax=fig.add_subplot(gs[1,2:4]);cv=sm['Cute'][bp];ev=sm['Erotic'][bp]
    ax.scatter(cv,ev,s=8,alpha=0.3,color='#27ae60');mn_=min(min(cv),min(ev));mx_=max(max(cv),max(ev));ax.plot([mn_,mx_],[mn_,mx_],'k--')
    ax.set_xlabel('Cute');ax.set_ylabel('Erotic');ax.set_title(f'Within-Pos: {bp}\nd={ar[bp]["d_pos"]:.4f}, p={ar[bp]["p_pos"]:.1e}',fontsize=10,fontweight='bold');ax.grid(alpha=0.2)
    # Boxplots for top 4 cross-valence
    top4=sorted(VALIDATED,key=lambda m:abs(ar[m]['d_cross']),reverse=True)[:4]
    for pi,met in enumerate(top4):
        ax=fig.add_subplot(gs[2,pi]);data=[sm[cat][met] for cat in CATS]
        bp_=ax.boxplot(data,patch_artist=True,tick_labels=['Thr','Mut','Cut','Ero'])
        for patch,cat in zip(bp_['boxes'],CATS):patch.set_facecolor(CCOL[cat]);patch.set_alpha(0.6)
        ax.set_title(f'{met[:15]}\nd_cross={ar[met]["d_cross"]:.3f}',fontsize=9,fontweight='bold');ax.grid(axis='y',alpha=0.1)
    fig.suptitle('Analysis 6.4: Affective Subcategory Dissociation\n211 subjects x 4 categories x 9 validated metrics',fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/analysis6_4f_dissociation.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  analysis6_4f ({time.time()-t0:.0f}s)")

    pickle.dump({'all_metrics':all_mets,'subj_metrics':sm,'analysis_results':ar,
                 'validated_metrics':VALIDATED,'subjects':subjs,'categories':CATS,'channels':a.channels},
                open(f'{a.results_dir}/ch6_exp4_full.pkl','wb'))
    print(f"\nDone. 6 figures. {time.time()-t0:.0f}s.")

if __name__=='__main__':main()
