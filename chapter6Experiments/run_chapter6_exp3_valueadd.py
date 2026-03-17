#!/usr/bin/env python3
"""
run_chapter6_exp3_valueadd.py — ARSPI-Net Chapter 6, Experiment 6.3
Nonlinear Transformation Value-Add: Raw EEG vs Reservoir Descriptors

Scientific Cycle: Motivation → Design → Raw Observation → Analysis → Conclusion

MATHEMATICAL MOTIVATION:
  The reservoir performs u(t) in R -> {M(t), S(t)} in R^256 x {0,1}^256.
  Cover's theorem (1965) predicts nonlinear projections into higher-dimensional
  spaces can make previously inseparable patterns separable. The data processing
  inequality constrains what survives. ENGINEERING QUESTION: does this expansion
  enhance temporal descriptor detectability on real clinical EEG, or does the
  raw signal already carry the same information in accessible form?

EXPERIMENTAL DESIGN:
  For 3 comparable temporal descriptors (permutation entropy, autocorrelation
  decay, LZ complexity), compute the descriptor BOTH on the raw EEG and on
  the reservoir output. Same mathematical functional. Different input spaces.
  211 subjects x 4 categories x 3 channels = 2,532 paired observations.

RAW OBSERVATION FIGURES (4):
  raw6_3_transformation_observation.pdf — The transformation itself: raw EEG,
      reservoir membrane/spikes, ordinal patterns on both, individual subjects
  raw6_3_population_distributions.pdf  — 211-subject histograms: EEG vs reservoir
  raw6_3_paired_scatter.pdf            — Subject-level EEG vs reservoir scatter
  raw6_3_paired_differences.pdf        — Within-valence paired difference histograms

ANALYSIS FIGURE (1):
  analysis6_3_value_add.pdf            — Effect size bar chart + summary table

Results:
  Permutation entropy: EEG d=+0.04, Reservoir d=-0.29 (6.8x gain). The reservoir
  creates detectability where none exists in the raw signal.
  tau_AC within-neg: EEG d=-0.25, Reservoir d=-0.15 (0.59x). Raw EEG is stronger
  for amplitude-dominated contrasts.
  Reservoir-only metrics (total_spikes, rate_entropy): d~0.22, p<0.001. New
  observables with no linear analogue.

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

N_RES=256; BETA=0.05; M_TH=0.5; EXCLUDED={127}
PAT=re.compile(r'SHAPE_Community_(\d+)_IAPS(Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC\.txt')
CATS=['Threat','Mutilation','Cute','Erotic']
CCOL={'Threat':'#e74c3c','Mutilation':'#c0392b','Cute':'#27ae60','Erotic':'#2ecc71'}
COMPARABLE=['permutation_entropy','tau_ac','lz_complexity']
COMP_LABELS=['Permutation Entropy (H_pi)','Autocorrelation Decay (tau_AC)','Lempel-Ziv Complexity']

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

def lz_complex(x_bin):
    n=len(x_bin);vocab=set();c=0;w=''
    for x in x_bin:
        wc=w+str(int(x))
        if wc not in vocab:vocab.add(wc);c+=1;w=''
        else:w=wc
    if w:c+=1
    return float(c/(n/(np.log2(n)+1e-10)))

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

def smeans(md,met,subjects,cats,channels):
    out={cat:[] for cat in cats}
    for sid in subjects:
        for cat in cats:
            vals=[md[(sid,cat,ch)][met] for ch in channels if (sid,cat,ch) in md and met in md[(sid,cat,ch)]]
            if vals:out[cat].append(np.mean(vals))
    return out

def main():
    p=argparse.ArgumentParser(description='Exp 6.3: Nonlinear Transformation Value-Add')
    p.add_argument('--category-dirs',nargs='+',required=True)
    p.add_argument('--output-dir',default='figures');p.add_argument('--results-dir',default='results')
    p.add_argument('--channels',nargs='+',type=int,default=[0,16,33])
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();os.makedirs(a.output_dir,exist_ok=True);os.makedirs(a.results_dir,exist_ok=True)
    t0=time.time();af=load_files(a.category_dirs);subjs=sorted(set(s for s,c in af.keys()))
    reservoir=Res(seed=a.seed)
    print(f"Loaded {len(af)} files, {len(subjs)} subjects")

    # ══════════════════════════════════════════════════════════
    # RAW OBSERVATION FIG 1: The transformation itself
    # ══════════════════════════════════════════════════════════
    dsid=subjs[len(subjs)//4];dch=16
    eeg=np.loadtxt(af[(dsid,'Threat')]);u_raw=norm(eeg[:,dch])
    M_d,S_d=reservoir.run(u_raw);mV=M_d.mean(1);pr=S_d.mean(1).astype(float)

    fig=plt.figure(figsize=(20,22));gs=GridSpec(6,2,figure=fig,hspace=0.55,wspace=0.35)
    # Row 1: raw input
    ax=fig.add_subplot(gs[0,:]);t_ms=np.arange(len(u_raw))/1.024
    ax.plot(t_ms,u_raw,'k-',lw=0.5);ax.set_ylabel('Amplitude');ax.grid(alpha=0.1)
    ax.set_title(f'The Input: Raw EEG (S{dsid}, Ch{dch}, Threat). T=1229 at 1024 Hz.',fontsize=12,fontweight='bold')
    # Row 2: reservoir output
    ax=fig.add_subplot(gs[1,0]);ax.plot(t_ms,mV,'b-',lw=0.5);ax.set_ylabel('Mean Membrane V')
    ax.set_title('Reservoir Mean Membrane (256 neurons avg)',fontsize=10,fontweight='bold');ax.grid(alpha=0.1)
    ax=fig.add_subplot(gs[1,1]);ax.plot(t_ms,uniform_filter1d(pr,20),'r-',lw=0.8);ax.set_ylabel('Pop Rate')
    ax.set_title('Reservoir Population Firing Rate',fontsize=10,fontweight='bold');ax.grid(alpha=0.1)
    # Row 3: ordinal patterns on both
    seg=slice(300,600)
    ax=fig.add_subplot(gs[2,0]);ax.plot(range(300),u_raw[seg],'k-',lw=1.0)
    for t in range(0,280,20):ax.plot(range(t,t+4),u_raw[300+t:300+t+4],'o-',ms=4,lw=1.5,alpha=0.6)
    ax.set_title('Raw EEG: Ordinal Patterns (d=4)',fontsize=10,fontweight='bold')
    ax=fig.add_subplot(gs[2,1]);mv_s=mV[seg];ax.plot(range(300),mv_s,'b-',lw=1.0)
    for t in range(0,280,20):ax.plot(range(t,t+4),mv_s[t:t+4],'o-',ms=4,lw=1.5,alpha=0.6)
    ax.set_title('Reservoir Membrane: Ordinal Patterns (d=4)',fontsize=10,fontweight='bold')
    # Row 4: metric values for one subject, all 4 cats
    ax=fig.add_subplot(gs[3,0]);x_=np.arange(4);w_=0.35
    ehpi={};rhpi={}
    for cat in CATS:
        ec=np.loadtxt(af[(dsid,cat)]);uc=norm(ec[:,dch]);Mc,_=reservoir.run(uc)
        ehpi[cat]=perm_entropy(uc);rhpi[cat]=perm_entropy(Mc.mean(1))
    ax.bar(x_-w_/2,[ehpi[c] for c in CATS],w_,color='#95a5a6',edgecolor='black',lw=0.5,label='Raw EEG')
    ax.bar(x_+w_/2,[rhpi[c] for c in CATS],w_,color='#3498db',edgecolor='black',lw=0.5,label='Reservoir')
    ax.set_xticks(x_);ax.set_xticklabels(CATS,fontsize=9);ax.set_ylabel('H_pi');ax.legend(fontsize=8)
    ax.set_title(f'H_pi: One Subject (S{dsid})',fontsize=10,fontweight='bold')
    for i,cat in enumerate(CATS):
        ax.text(i-w_/2,ehpi[cat]+0.002,f'{ehpi[cat]:.4f}',ha='center',fontsize=6)
        ax.text(i+w_/2,rhpi[cat]+0.002,f'{rhpi[cat]:.4f}',ha='center',fontsize=6)
    # Same for tau_ac
    ax=fig.add_subplot(gs[3,1])
    etac={};rtac={}
    for cat in CATS:
        ec=np.loadtxt(af[(dsid,cat)]);uc=norm(ec[:,dch]);_,Sc=reservoir.run(uc)
        etac[cat]=ac_decay(uc);rtac[cat]=ac_decay(Sc.mean(1).astype(float))
    ax.bar(x_-w_/2,[etac[c] for c in CATS],w_,color='#95a5a6',edgecolor='black',lw=0.5,label='Raw EEG')
    ax.bar(x_+w_/2,[rtac[c] for c in CATS],w_,color='#3498db',edgecolor='black',lw=0.5,label='Reservoir')
    ax.set_xticks(x_);ax.set_xticklabels(CATS,fontsize=9);ax.set_ylabel('tau_AC');ax.legend(fontsize=8)
    ax.set_title(f'tau_AC: One Subject (S{dsid})',fontsize=10,fontweight='bold')
    # Row 5: 6 individual subject profiles
    ax=fig.add_subplot(gs[4,:]);psids=[subjs[i] for i in [10,40,80,120,160,200]]
    bxe=[];bxr=[];be=[];br=[];bc=[];tp=[];tl=[];off=0
    for sid in psids:
        for ci,cat in enumerate(CATS):
            ec=np.loadtxt(af[(sid,cat)]);uc=norm(ec[:,dch]);Mc,_=reservoir.run(uc)
            he=perm_entropy(uc);hr=perm_entropy(Mc.mean(1))
            bxe.append(off);bxr.append(off+0.35);be.append(he);br.append(hr);bc.append(CCOL[cat]);off+=0.8
        tp.append(off-2.0);tl.append(f'S{sid}');off+=0.8
    ax.bar(bxe,be,0.3,color='#95a5a6',edgecolor='black',lw=0.2)
    ax.bar(bxr,br,0.3,color=bc,edgecolor='black',lw=0.2,alpha=0.7)
    ax.set_xticks(tp);ax.set_xticklabels(tl,fontsize=8);ax.set_ylabel('H_pi')
    ax.set_title('6 Subjects x 4 Categories: Gray=Raw EEG, Colored=Reservoir',fontsize=10,fontweight='bold')
    # Row 6: autocorrelation
    ax=fig.add_subplot(gs[5,0])
    for cat,col in [('Threat','#e74c3c'),('Cute','#27ae60')]:
        uc=norm(np.loadtxt(af[(dsid,cat)])[:,dch]);xc=uc-uc.mean();v=np.sum(xc**2)
        ax.plot([np.sum(xc[:len(uc)-k]*xc[k:])/v for k in range(100)],color=col,lw=1.5,label=f'{cat}')
    ax.axhline(np.exp(-1),color='gray',ls=':');ax.set_xlabel('Lag');ax.set_ylabel('AC')
    ax.set_title(f'Raw EEG Autocorrelation (S{dsid})',fontsize=10,fontweight='bold');ax.legend(fontsize=8)
    ax=fig.add_subplot(gs[5,1])
    for cat,col in [('Threat','#e74c3c'),('Cute','#27ae60')]:
        uc=norm(np.loadtxt(af[(dsid,cat)])[:,dch]);_,Sc=reservoir.run(uc);pc=Sc.mean(1).astype(float)
        xc=pc-pc.mean();v=np.sum(xc**2)
        if v>0:ax.plot([np.sum(xc[:len(pc)-k]*xc[k:])/v for k in range(100)],color=col,lw=1.5,ls='--',label=f'{cat}')
    ax.axhline(np.exp(-1),color='gray',ls=':');ax.set_xlabel('Lag');ax.set_ylabel('AC')
    ax.set_title(f'Reservoir Autocorrelation (S{dsid})',fontsize=10,fontweight='bold');ax.legend(fontsize=8)
    fig.suptitle('Raw Observation: The Nonlinear State Expansion\nThe reader sees the transformation and its consequences before any statistical comparison.',fontsize=13,fontweight='bold',y=1.02)
    plt.savefig(f'{a.output_dir}/raw6_3_transformation_observation.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_3 fig1 ({time.time()-t0:.0f}s)")

    # ══════════════════════════════════════════════════════════
    # COMPUTE PAIRED METRICS (ALL SUBJECTS)
    # ══════════════════════════════════════════════════════════
    print("  Computing paired metrics...")
    em={};rm={}
    for si,sid in enumerate(subjs):
        if si%50==0:print(f"    {si+1}/{len(subjs)}... {time.time()-t0:.0f}s")
        for cat in CATS:
            if (sid,cat) not in af:continue
            eeg_=np.loadtxt(af[(sid,cat)])
            for ch in a.channels:
                u=norm(eeg_[:,ch])
                em[(sid,cat,ch)]={'permutation_entropy':perm_entropy(u),'tau_ac':ac_decay(u),
                    'lz_complexity':lz_complex((u>np.median(u)).astype(int)),'signal_variance':float(np.var(u))}
                M,S=reservoir.run(u);pr_=S.mean(1).astype(float);pnr=S.mean(0)
                neurons=np.random.RandomState(0).choice(N_RES,10,replace=False);lzv=[]
                for ni in neurons:
                    sq=S[:,ni];n=len(sq);vocab=set();c_=0;w=''
                    for x_ in sq:
                        wc=w+str(int(x_))
                        if wc not in vocab:vocab.add(wc);c_+=1;w=''
                        else:w=wc
                    if w:c_+=1
                    lzv.append(c_/(n/(np.log2(n)+1e-10)))
                rm[(sid,cat,ch)]={'permutation_entropy':perm_entropy(M.mean(1)),'tau_ac':ac_decay(pr_),
                    'lz_complexity':float(np.mean(lzv)),'total_spikes':float(S.sum()),
                    'rate_entropy':float(-np.sum(pnr*np.log2(pnr+1e-15)+(1-pnr)*np.log2(1-pnr+1e-15))/N_RES),
                    'rate_variance':float(np.var(pnr))}

    # ══════════════════════════════════════════════════════════
    # RAW OBSERVATION FIG 2: Population distributions
    # ══════════════════════════════════════════════════════════
    fig,axes=plt.subplots(3,4,figsize=(22,16))
    for mi,met in enumerate(COMPARABLE):
        esm=smeans(em,met,subjs,CATS,a.channels);rsm=smeans(rm,met,subjs,CATS,a.channels)
        for ci,cat in enumerate(CATS):
            ax=axes[mi,ci];ev=esm[cat];rv=rsm[cat]
            ax.hist(ev,bins=25,alpha=0.5,color='#95a5a6',edgecolor='black',lw=0.3,density=True,label='Raw EEG')
            ax.hist(rv,bins=25,alpha=0.5,color=CCOL[cat],edgecolor='black',lw=0.3,density=True,label='Reservoir')
            ax.axvline(np.mean(ev),color='gray',lw=2,ls='--');ax.axvline(np.mean(rv),color=CCOL[cat],lw=2,ls='-')
            if ci==0:ax.set_ylabel(COMP_LABELS[mi][:15],fontsize=9,fontweight='bold')
            if mi==0:ax.set_title(cat,fontsize=11,fontweight='bold',color=CCOL[cat])
            if mi==0 and ci==0:ax.legend(fontsize=7)
            ax.text(0.98,0.95,f'EEG:{np.mean(ev):.4f}\nRes:{np.mean(rv):.4f}',transform=ax.transAxes,fontsize=7,ha='right',va='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
    fig.suptitle('Raw Observation: Population Distributions (N=211)\nGray=Raw EEG. Colored=Reservoir.',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_3_population_distributions.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_3 fig2 ({time.time()-t0:.0f}s)")

    # ══════════════════════════════════════════════════════════
    # RAW OBSERVATION FIG 3: Paired scatter
    # ══════════════════════════════════════════════════════════
    fig,axes=plt.subplots(3,4,figsize=(22,16))
    for mi,met in enumerate(COMPARABLE):
        esm=smeans(em,met,subjs,CATS,a.channels);rsm=smeans(rm,met,subjs,CATS,a.channels)
        for ci,cat in enumerate(CATS):
            ax=axes[mi,ci];ev=esm[cat];rv=rsm[cat];n_=min(len(ev),len(rv))
            ax.scatter(ev[:n_],rv[:n_],s=5,alpha=0.4,color=CCOL[cat])
            r_=np.corrcoef(ev[:n_],rv[:n_])[0,1]
            if ci==0:ax.set_ylabel(f'{COMP_LABELS[mi][:10]}\nReservoir',fontsize=9,fontweight='bold')
            if mi==0:ax.set_title(cat,fontsize=11,fontweight='bold',color=CCOL[cat])
            ax.text(0.05,0.95,f'r={r_:.3f}',transform=ax.transAxes,fontsize=9,fontweight='bold',va='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
            ax.grid(alpha=0.1)
    fig.suptitle('Raw Observation: Paired Scatter — Raw EEG (x) vs Reservoir (y)\nHigh r: reservoir tracks raw. Low r: different representation.',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_3_paired_scatter.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_3 fig3 ({time.time()-t0:.0f}s)")

    # ══════════════════════════════════════════════════════════
    # RAW OBSERVATION FIG 4: Within-valence paired differences
    # ══════════════════════════════════════════════════════════
    contrasts=[('Threat','Mutilation','Within-Neg','#e74c3c'),('Cute','Erotic','Within-Pos','#27ae60')]
    fig,axes=plt.subplots(3,2,figsize=(16,14))
    for mi,met in enumerate(COMPARABLE):
        esm=smeans(em,met,subjs,CATS,a.channels);rsm=smeans(rm,met,subjs,CATS,a.channels)
        for ci,(c1,c2,label,color) in enumerate(contrasts):
            ax=axes[mi,ci]
            de=np.array(esm[c1])-np.array(esm[c2]);dr=np.array(rsm[c1])-np.array(rsm[c2])
            ax.hist(de,bins=30,alpha=0.5,color='#95a5a6',edgecolor='black',lw=0.3,density=True,label=f'EEG (mean={de.mean():.4f})')
            ax.hist(dr,bins=30,alpha=0.5,color=color,edgecolor='black',lw=0.3,density=True,label=f'Res (mean={dr.mean():.4f})')
            ax.axvline(0,color='black',lw=1.5);ax.axvline(de.mean(),color='gray',lw=2,ls='--');ax.axvline(dr.mean(),color=color,lw=2,ls='-')
            if ci==0:ax.set_ylabel(COMP_LABELS[mi][:15],fontsize=9,fontweight='bold')
            if mi==0:ax.set_title(f'{label}: {c1} - {c2}',fontsize=11,fontweight='bold')
            ax.legend(fontsize=7);ax.grid(alpha=0.1)
    fig.suptitle('Raw Observation: Within-Valence Paired Differences\nGray=Raw EEG. Colored=Reservoir. Shift from zero = the signal.',fontsize=13,fontweight='bold',y=1.03)
    plt.tight_layout();plt.savefig(f'{a.output_dir}/raw6_3_paired_differences.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_3 fig4 ({time.time()-t0:.0f}s)")

    # ══════════════════════════════════════════════════════════
    # ANALYSIS (comes LAST)
    # ══════════════════════════════════════════════════════════
    print("\n  Analysis:")
    cr={}
    for met in COMPARABLE:
        esm=smeans(em,met,subjs,CATS,a.channels);rsm=smeans(rm,met,subjs,CATS,a.channels);cr[met]={}
        for c1,c2,label,_ in contrasts:
            e1=np.array(esm[c1]);e2=np.array(esm[c2]);r1=np.array(rsm[c1]);r2=np.array(rsm[c2])
            de_=(e1-e2);d_e=de_.mean()/(de_.std()+1e-15);dr_=(r1-r2);d_r=dr_.mean()/(dr_.std()+1e-15)
            try:_,pe=wilcoxon(e1,e2)
            except:pe=1.0
            try:_,pr_=wilcoxon(r1,r2)
            except:pr_=1.0
            gain=abs(d_r)/(abs(d_e)+1e-10)
            cr[met][label]={'d_eeg':float(d_e),'p_eeg':float(pe),'d_res':float(d_r),'p_res':float(pr_),'gain':float(gain)}
            print(f"    {met:<20} {label:<12} d_EEG={d_e:+.4f} d_Res={d_r:+.4f} gain={gain:.2f}x")

    fig=plt.figure(figsize=(20,10));gs=GridSpec(1,3,figure=fig,wspace=0.4)
    ax=fig.add_subplot(gs[0,0:2]);al=[];ade=[];adr=[]
    for met in COMPARABLE:
        for label in ['Within-Neg','Within-Pos']:
            r=cr[met][label];al.append(f'{met[:10]}\n{label[:7]}');ade.append(abs(r['d_eeg']));adr.append(abs(r['d_res']))
    x_=np.arange(len(al));w_=0.35
    ax.bar(x_-w_/2,ade,w_,color='#95a5a6',edgecolor='black',lw=0.5,label='Raw EEG')
    ax.bar(x_+w_/2,adr,w_,color='#3498db',edgecolor='black',lw=0.5,label='Reservoir')
    ax.set_xticks(x_);ax.set_xticklabels(al,fontsize=7);ax.set_ylabel("|Cohen's d|");ax.legend(fontsize=9)
    ax.set_title('Effect Size: Same Functional, Two Paths',fontsize=11,fontweight='bold');ax.grid(axis='y',alpha=0.2)
    ax=fig.add_subplot(gs[0,2]);ax.axis('off')
    td=[]
    for met in COMPARABLE:
        for label in ['Within-Neg','Within-Pos']:
            r=cr[met][label];td.append([met[:10],label[:7],f"{r['d_eeg']:+.3f}",f"{r['d_res']:+.3f}",f"{r['gain']:.1f}x"])
    tab=ax.table(cellText=td,colLabels=['Metric','Contr','d_EEG','d_Res','Gain'],loc='center',cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,1.8)
    for i in range(len(td)):
        g=float(td[i][-1].replace('x',''));c='#d5f5e3' if g>1 else '#fadbd8'
        for j in range(5):tab[i+1,j].set_facecolor(c)
    ax.set_title('Verdict',fontsize=11,fontweight='bold',pad=15)
    fig.suptitle('Analysis: The Engineering Verdict',fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/analysis6_3_value_add.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  analysis ({time.time()-t0:.0f}s)")

    pickle.dump({'comparison_results':cr,'comparable_metrics':COMPARABLE,
                 'contrasts':[(c1,c2,l) for c1,c2,l,_ in contrasts]},
                open(f'{a.results_dir}/ch6_exp3new_full.pkl','wb'))
    print(f"\nDone. 5 figures. {time.time()-t0:.0f}s.")

if __name__=='__main__':main()
