#!/usr/bin/env python3
"""
run_chapter6_exp3_surrogate.py — ARSPI-Net Chapter 6, Experiment 6.3
Surrogate Sensitivity / Null Discrimination

Scientific Cycle: Motivation → Design → Raw Observation → Analysis → Conclusion

Three surrogate families destroying different temporal structure levels:
  Phase-randomized — preserves amplitude spectrum, destroys phase relationships
  Time-shuffled    — destroys all temporal structure
  Block-shuffled   — preserves local structure (<50 steps), destroys global

Generates 2 figures:
  raw6_3a_surrogate_comparison.pdf — Real vs surrogates: input, spikes, dynamics,
                                      phase portraits, power spectra, autocorrelation (20 panels)
  analysis6_3b_surrogate_gate.pdf  — Cohen's d heatmap + gate decision

Results: 9/11 pass (sensitive to >=2/3 types). tau_AC and LZ pass all 3.
         Temporal structure metrics detect genuine phase-dependent organization.

Usage:
    python run_chapter6_exp3_surrogate.py \
        --category-dirs categoriesbatch1 ... --output-dir figures/ --results-dir results/

Author: Andrew (ARSPI-Net Dissertation), March 2026
"""
import numpy as np,os,re,pickle,time,math,argparse
from collections import defaultdict
from scipy.stats import wilcoxon
import matplotlib;matplotlib.use('Agg')
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
SURR_TYPES=['phase_randomized','time_shuffled','block_shuffled']
SURR_LABELS=['Phase-Rand','Time-Shuff','Block-Shuff']
SURR_COLORS=['#3498db','#e74c3c','#f39c12']

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
    if nc>0:p_=np.array(list(pats.values()))/nc;mets['permutation_entropy']=float(-np.sum(p_*np.log2(p_+1e-15))/np.log2(math.factorial(d)))
    else:mets['permutation_entropy']=0.0
    ps=np.convolve(pr,np.ones(20)/20,mode='same');tp=np.argmax(ps[100:])+100
    if tp<T-50:
        dec=ps[tp:];rp=dec[0];ri=dec[-50:].mean()
        if rp>ri+0.001:tgt=ri+(rp-ri)/np.e;idx=np.where(dec<=tgt)[0];mets['tau_relax']=float(idx[0]) if len(idx)>0 else float(len(dec))
        else:mets['tau_relax']=0.0
    else:mets['tau_relax']=0.0
    pc=pr-pr.mean();vp=np.sum(pc**2)
    if vp>0:ml=min(200,T//2);ac=np.array([np.sum(pc[:T-k]*pc[k:])/vp for k in range(ml)]);idx=np.where(ac<=np.exp(-1))[0];mets['tau_ac']=float(idx[0]) if len(idx)>0 else float(ml)
    else:mets['tau_ac']=0.0
    return mets

def phase_randomize(sig,rng):
    fft=np.fft.rfft(sig);n=len(sig);ph=rng.uniform(0,2*np.pi,len(fft));ph[0]=0
    if n%2==0:ph[-1]=0
    return np.fft.irfft(np.abs(fft)*np.exp(1j*ph),n=n)
def time_shuffle(sig,rng):return sig[rng.permutation(len(sig))]
def block_shuffle(sig,rng,bs=50):
    n=len(sig);nb=n//bs;bl=[sig[i*bs:(i+1)*bs] for i in range(nb)];rem=sig[nb*bs:];rng.shuffle(bl);return np.concatenate(bl+[rem])
def norm(u):return(u-u.mean())/(u.std()+1e-10)

def load_files(dirs):
    af={}
    for d in dirs:
        if not os.path.isdir(d):continue
        for f in os.listdir(d):
            m=PAT.match(f)
            if m:sid=int(m.group(1));cat=m.group(3);(af.__setitem__((sid,cat),os.path.join(d,f)) if sid not in EXCLUDED else None)
    return af

def main():
    p=argparse.ArgumentParser(description='Exp 6.3: Surrogate Sensitivity')
    p.add_argument('--category-dirs',nargs='+',required=True)
    p.add_argument('--output-dir',default='figures');p.add_argument('--results-dir',default='results')
    p.add_argument('--n-subjects',type=int,default=40);p.add_argument('--channels',nargs='+',type=int,default=[0,16,33])
    a=p.parse_args();os.makedirs(a.output_dir,exist_ok=True);os.makedirs(a.results_dir,exist_ok=True)
    t0=time.time();af=load_files(a.category_dirs);subjs=sorted(set(s for s,c in af.keys()))
    reservoir=Res(42);dsid=subjs[len(subjs)//4];dch=16

    # ── FIG 1: What surrogates look like (20 panels) ──
    eeg=np.loadtxt(af[(dsid,'Threat')]);u_real=norm(eeg[:,dch])
    rng_s=np.random.RandomState(42)
    u_ph=norm(phase_randomize(u_real,rng_s));u_ts=norm(time_shuffle(u_real,np.random.RandomState(43)));u_bl=norm(block_shuffle(u_real,np.random.RandomState(44)))
    sigs=[('Real',u_real,'#2c3e50'),('Phase-Rand',u_ph,'#3498db'),('Time-Shuff',u_ts,'#e74c3c'),('Block-Shuff',u_bl,'#f39c12')]

    fig=plt.figure(figsize=(20,20));gs=GridSpec(5,4,figure=fig,hspace=0.5,wspace=0.35)
    Mo={};So={}
    for si,(lab,u,col) in enumerate(sigs):
        ax=fig.add_subplot(gs[0,si]);ax.plot(u[:600],color=col,lw=0.5);ax.set_ylim(-4,4);ax.set_title(f'{lab}',fontsize=10,fontweight='bold',color=col);ax.grid(alpha=0.1)
        M,S=reservoir.run(u);Mo[lab]=M;So[lab]=S
        ax=fig.add_subplot(gs[1,si]);t_,n_=np.where(S[:,:60].T>0);ax.scatter(n_,t_,s=0.1,c=col,alpha=0.3,rasterized=True)
        ax.set_title(f'{int(S.sum()):,} spikes',fontsize=9,fontweight='bold')
        ax=fig.add_subplot(gs[2,si]);ax.plot(uniform_filter1d(S.mean(1).astype(float),20),color=col,lw=0.8);ax.set_ylim(0,0.35)
        ax=fig.add_subplot(gs[3,si]);z=PCA(2,random_state=42).fit_transform(M)
        cmaps=['Greys','Blues','Reds','Oranges'];ax.scatter(z[:,0],z[:,1],c=np.arange(len(z)),cmap=cmaps[si],s=0.5,alpha=0.4,rasterized=True)
        ax.set_title(f'Phase Portrait',fontsize=9,fontweight='bold')
    ax=fig.add_subplot(gs[4,0:2])
    for lab,u,col in sigs:
        fp=np.abs(np.fft.rfft(u))**2;fr=np.fft.rfftfreq(len(u),d=1/1024);ax.semilogy(fr[:200],fp[:200],color=col,lw=0.8,alpha=0.7,label=lab)
    ax.set_xlabel('Freq (Hz)');ax.set_title('Power Spectra',fontsize=10,fontweight='bold');ax.legend(fontsize=7)
    ax=fig.add_subplot(gs[4,2:4])
    for lab,u,col in sigs:
        ac_u=np.correlate(u[:500]-u[:500].mean(),u[:500]-u[:500].mean(),mode='full');ac_u=ac_u[len(ac_u)//2:len(ac_u)//2+100]/ac_u[len(ac_u)//2]
        ax.plot(ac_u,color=col,lw=1.0,label=lab)
    ax.set_xlabel('Lag');ax.set_title('Autocorrelation',fontsize=10,fontweight='bold');ax.legend(fontsize=7)
    fig.suptitle(f'Raw Observation 6.8: Real vs Surrogates (S{dsid}, Ch{dch})',fontsize=13,fontweight='bold',y=1.02)
    plt.savefig(f'{a.output_dir}/raw6_3a_surrogate_comparison.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  raw6_3a ({time.time()-t0:.0f}s)")

    # ── Full surrogate computation ──
    rng=np.random.RandomState(42);ssub=sorted(rng.choice(subjs,min(a.n_subjects,len(subjs)),replace=False))
    results={st:{m:{'real':[],'surr':[]} for m in METRICS} for st in SURR_TYPES}
    print(f"Computing: {len(ssub)} subjects x 4 cats x {len(a.channels)} ch x 3 surrogates")
    for si,sid in enumerate(ssub):
        if si%10==0:print(f"  {si+1}/{len(ssub)}... {time.time()-t0:.0f}s")
        for cat in CATS:
            if (sid,cat) not in af:continue
            eeg=np.loadtxt(af[(sid,cat)])
            for ch in a.channels:
                u=norm(eeg[:,ch]);M_r,S_r=reservoir.run(u);mets_r=compute_metrics(M_r,S_r,reservoir)
                rng_s=np.random.RandomState(sid*100+ch)
                for st in SURR_TYPES:
                    if st=='phase_randomized':u_s=norm(phase_randomize(u,rng_s))
                    elif st=='time_shuffled':u_s=norm(time_shuffle(u,rng_s))
                    else:u_s=norm(block_shuffle(u,rng_s))
                    M_s,S_s=reservoir.run(u_s);mets_s=compute_metrics(M_s,S_s,reservoir)
                    for mn in METRICS:results[st][mn]['real'].append(mets_r[mn]);results[st][mn]['surr'].append(mets_s[mn])

    # ── Analysis ──
    sens={}
    for st in SURR_TYPES:
        sens[st]={}
        for mn in METRICS:
            rl=np.array(results[st][mn]['real']);sr=np.array(results[st][mn]['surr'])
            pool=np.sqrt((rl.std()**2+sr.std()**2)/2);cd=(rl.mean()-sr.mean())/(pool+1e-15)
            try:_,pv=wilcoxon(rl,sr)
            except:pv=1.0
            sens[st][mn]={'cohen_d':float(cd),'p_value':float(pv),'sensitive':abs(cd)>0.2 and pv<0.05}

    # ── FIG 2: Gate decision ──
    fig=plt.figure(figsize=(20,12));gs=GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.4)
    ax=fig.add_subplot(gs[0,0:2]);dm=np.zeros((len(METRICS),len(SURR_TYPES)))
    for si,st in enumerate(SURR_TYPES):
        for mi,mn in enumerate(METRICS):dm[mi,si]=sens[st][mn]['cohen_d']
    im=ax.imshow(dm.T,aspect='auto',cmap='RdBu_r',vmin=-2,vmax=2)
    ax.set_xticks(range(len(METRICS)));ax.set_xticklabels(SHORT,rotation=45,ha='right',fontsize=8)
    ax.set_yticks(range(3));ax.set_yticklabels(SURR_LABELS,fontsize=9);plt.colorbar(im,ax=ax,shrink=0.6)
    for si,st in enumerate(SURR_TYPES):
        for mi,mn in enumerate(METRICS):
            if sens[st][mn]['sensitive']:ax.text(mi,si,'*',ha='center',va='center',fontsize=14,fontweight='bold')
    ax.set_title("Cohen's d (* = significant)",fontsize=11,fontweight='bold')
    ax=fig.add_subplot(gs[0,2]);ns=[sum(1 for st in SURR_TYPES if sens[st][mn]['sensitive']) for mn in METRICS]
    cols=['#27ae60' if n>=2 else '#f39c12' if n==1 else '#e74c3c' for n in ns]
    ax.barh(range(len(METRICS)),ns,color=cols,edgecolor='black');ax.set_yticks(range(len(METRICS)));ax.set_yticklabels(SHORT,fontsize=8)
    ax.axvline(2,color='green',ls='--',lw=1.5);ax.set_title('Gate (need 2/3)',fontsize=11,fontweight='bold')
    ax=fig.add_subplot(gs[1,0:2]);x=np.arange(len(METRICS));w=0.25
    for si,st in enumerate(SURR_TYPES):dv=[abs(sens[st][mn]['cohen_d']) for mn in METRICS];ax.bar(x+si*w,dv,w,color=SURR_COLORS[si],edgecolor='black',lw=0.3,label=SURR_LABELS[si])
    ax.set_xticks(x+w);ax.set_xticklabels(SHORT,rotation=45,ha='right',fontsize=8);ax.set_yscale('symlog',linthresh=0.5)
    ax.axhline(0.2,color='gray',ls=':');ax.legend(fontsize=7,ncol=2);ax.set_title('|d| by Type',fontsize=11,fontweight='bold')
    ax=fig.add_subplot(gs[1,2]);rt=np.array(results['time_shuffled']['tau_ac']['real']);st_=np.array(results['time_shuffled']['tau_ac']['surr'])
    ax.scatter(rt,st_,s=5,alpha=0.3,color='#e74c3c');mn_=min(rt.min(),st_.min());mx_=max(rt.max(),st_.max());ax.plot([mn_,mx_],[mn_,mx_],'k--')
    ax.set_xlabel('Real tau_AC');ax.set_ylabel('Time-Shuff');ax.set_title(f'tau_AC: d={sens["time_shuffled"]["tau_ac"]["cohen_d"]:.2f}',fontsize=10,fontweight='bold')
    n_pass=sum(1 for mn in METRICS if sum(1 for st in SURR_TYPES if sens[st][mn]['sensitive'])>=2)
    fig.suptitle(f'Analysis 6.3: Surrogate Gate — {n_pass}/11 Pass',fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/analysis6_3b_surrogate_gate.pdf',bbox_inches='tight',dpi=150);plt.close()
    print(f"  analysis6_3b ({time.time()-t0:.0f}s)")

    pickle.dump({'results':results,'sensitivity':sens,'surrogate_types':SURR_TYPES,'metric_names':METRICS},
                open(f'{a.results_dir}/ch6_exp3_full.pkl','wb'))
    passed_s=[mn for mn in METRICS if sum(1 for st in SURR_TYPES if sens[st][mn]['sensitive'])>=2]
    print(f"\nDone. 2 figures. {time.time()-t0:.0f}s. Passed: {passed_s}")

if __name__=='__main__':main()
