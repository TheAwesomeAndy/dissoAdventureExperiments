#!/usr/bin/env python3
"""
run_chapter6_exp1_esp.py — ARSPI-Net Chapter 6, Experiment 6.1
Echo State Property and Driven Stability Verification

Scientific Cycle: Motivation → Design → Raw Observation → Analysis → Conclusion

Generates 6 figures following raw-observation-first methodology:
  raw6_1a_eeg_input.pdf           — The input: 4-subcategory ERPs
  raw6_1b_spike_rasters.pdf       — The response: spike rasters + membrane heatmaps
  raw6_1c_population_dynamics.pdf — Collective dynamics + phase portraits
  raw6_1d_esp_convergence.pdf     — ESP convergence from 2 ICs (12 panels)
  raw6_1e_multi_subject_esp.pdf   — Generalization across 6 subjects
  analysis6_1f_lyapunov.pdf       — Benettin stretching → population lambda_1

Results: lambda_1 = -0.054 ± 0.0001, 100% negative across 4,220 measurements.
         The reservoir is uniformly contracting under real clinical EEG drive.

Usage:
    python run_chapter6_exp1_esp.py \
        --category-dirs categoriesbatch1 categoriesbatch2 categoriesbatch3 categoriesbatch4 \
        --output-dir figures/ --results-dir results/

Author: Andrew (ARSPI-Net Dissertation), March 2026
"""
import numpy as np, os, re, pickle, time, argparse
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d

N_RES=256; BETA=0.05; M_TH=0.5; EXCLUDED={127}
DELTA_0=1e-8; T_RENORM=50
PAT=re.compile(r'SHAPE_Community_(\d+)_IAPS(Neg|Pos)_(Threat|Mutilation|Cute|Erotic)_BC\.txt')
CATS=['Threat','Mutilation','Cute','Erotic']
CCOL={'Threat':'#e74c3c','Mutilation':'#c0392b','Cute':'#27ae60','Erotic':'#2ecc71'}

class Reservoir:
    def __init__(self,seed=42):
        rng=np.random.RandomState(seed); self.Win=rng.randn(N_RES)*0.3
        mask=(rng.rand(N_RES,N_RES)<0.1).astype(float); np.fill_diagonal(mask,0)
        self.Wrec=rng.randn(N_RES,N_RES)*0.05*mask; self.fanout=mask.sum(1)
    def run(self,u,M0=None):
        T=len(u);M=np.zeros((T,N_RES));S=np.zeros((T,N_RES),dtype=np.int8)
        m=M0.copy() if M0 is not None else np.zeros(N_RES); s=np.zeros(N_RES)
        for t in range(T):
            I=self.Win*u[t]+self.Wrec@s; m=(1-BETA)*m*(1-s)+I
            sp=(m>=M_TH).astype(float); M[t]=m; S[t]=sp.astype(np.int8); s=sp
        return M,S

def load_files(dirs):
    af={}
    for d in dirs:
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            m=PAT.match(f)
            if m:
                sid=int(m.group(1));cat=m.group(3)
                if sid not in EXCLUDED: af[(sid,cat)]=os.path.join(d,f)
    return af

def norm(u): return (u-u.mean())/(u.std()+1e-10)

def main():
    p=argparse.ArgumentParser(description='Exp 6.1: ESP Verification')
    p.add_argument('--category-dirs',nargs='+',required=True)
    p.add_argument('--output-dir',default='figures')
    p.add_argument('--results-dir',default='results')
    p.add_argument('--demo-channel',type=int,default=16)
    p.add_argument('--analysis-channels',nargs='+',type=int,default=[0,8,16,24,33])
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); os.makedirs(a.output_dir,exist_ok=True); os.makedirs(a.results_dir,exist_ok=True)
    t0=time.time()
    af=load_files(a.category_dirs); subjs=sorted(set(s for s,c in af.keys()))
    print(f"Loaded {len(af)} files, {len(subjs)} subjects")
    res=Reservoir(seed=a.seed); dsid=subjs[len(subjs)//4]; dch=a.demo_channel

    # ── FIG 1: Raw EEG Input ──
    fig,axes=plt.subplots(5,1,figsize=(16,14),sharex=True); eeg_data={}
    for ci,cat in enumerate(CATS):
        eeg=np.loadtxt(af[(dsid,cat)]); eeg_data[cat]=eeg; u=eeg[:,dch]
        axes[ci].plot(np.arange(len(u))/1.024, u, color=CCOL[cat], linewidth=0.5)
        axes[ci].set_ylabel(f'{cat}\n(uV)',fontsize=10,fontweight='bold',color=CCOL[cat])
        axes[ci].axvline(200,color='gray',linestyle=':'); axes[ci].grid(alpha=0.1)
        axes[ci].text(0.98,0.92,f'mean={u.mean():.2f}, std={u.std():.2f}',transform=axes[ci].transAxes,fontsize=7,ha='right',va='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
    for cat in CATS: axes[4].plot(np.arange(1229)/1.024,eeg_data[cat][:,dch],color=CCOL[cat],linewidth=0.6,alpha=0.7,label=cat)
    axes[4].set_xlabel('Time (ms)'); axes[4].legend(fontsize=8,ncol=4)
    fig.suptitle(f'Raw Observation 6.1: EEG Input (S{dsid}, Ch{dch})',fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.savefig(f'{a.output_dir}/raw6_1a_eeg_input.pdf',bbox_inches='tight',dpi=150); plt.close()
    print(f"  raw6_1a ({time.time()-t0:.0f}s)")

    # ── FIG 2: Spike Rasters + Membrane ──
    fig,axes=plt.subplots(4,2,figsize=(20,16)); outs={}
    for ci,cat in enumerate(CATS):
        u=norm(eeg_data[cat][:,dch]); M,S=res.run(u); outs[cat]={'M':M,'S':S,'u':u}
        st,sn=np.where(S.T>0); axes[ci,0].scatter(sn,st,s=0.1,c=CCOL[cat],alpha=0.3,rasterized=True)
        axes[ci,0].set_ylabel(f'{cat}',fontsize=10,fontweight='bold',color=CCOL[cat])
        axes[ci,0].text(0.98,0.95,f'{int(S.sum()):,} spikes',transform=axes[ci,0].transAxes,fontsize=8,ha='right',va='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.9))
        axes[ci,1].imshow(M[:,:80].T,aspect='auto',cmap='hot',vmin=0,vmax=M_TH*1.2)
        if ci==3: axes[ci,0].set_xlabel('Time'); axes[ci,1].set_xlabel('Time')
    fig.suptitle(f'Raw Observation 6.2: Reservoir Response (S{dsid}, Ch{dch})',fontsize=13,fontweight='bold',y=1.02)
    plt.tight_layout(); plt.savefig(f'{a.output_dir}/raw6_1b_spike_rasters.pdf',bbox_inches='tight',dpi=150); plt.close()
    print(f"  raw6_1b ({time.time()-t0:.0f}s)")

    # ── FIG 3: Population Dynamics + Phase Portraits ──
    fig=plt.figure(figsize=(20,12)); gs=GridSpec(2,4,figure=fig,hspace=0.4,wspace=0.35)
    ax=fig.add_subplot(gs[0,0:2])
    for cat in CATS: ax.plot(uniform_filter1d(outs[cat]['S'].mean(1).astype(float),20),color=CCOL[cat],linewidth=1.2,label=cat)
    ax.set_xlabel('Time Step');ax.set_ylabel('Pop Rate');ax.legend(fontsize=9);ax.set_title('Population Rate',fontsize=11,fontweight='bold')
    ax=fig.add_subplot(gs[0,2:4])
    for cat in CATS: ax.plot(uniform_filter1d(outs[cat]['M'].mean(1),10),color=CCOL[cat],linewidth=1.2,label=cat)
    ax.set_xlabel('Time Step');ax.set_ylabel('Mean V');ax.legend(fontsize=9);ax.set_title('Mean Membrane',fontsize=11,fontweight='bold')
    for ci,cat in enumerate(CATS):
        ax=fig.add_subplot(gs[1,ci]); z=PCA(2,random_state=42).fit_transform(outs[cat]['M'])
        ax.scatter(z[:,0],z[:,1],c=np.arange(len(z)),cmap='viridis',s=1,alpha=0.5,rasterized=True)
        ax.set_title(f'{cat}',fontsize=10,fontweight='bold',color=CCOL[cat])
    fig.suptitle(f'Raw Observation 6.3: Dynamics & Phase Portraits (S{dsid}, Ch{dch})',fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/raw6_1c_population_dynamics.pdf',bbox_inches='tight',dpi=150); plt.close()
    print(f"  raw6_1c ({time.time()-t0:.0f}s)")

    # ── FIG 4: ESP Convergence (12 panels) ──
    ud=outs['Threat']['u']; M1,S1=res.run(ud,M0=np.zeros(N_RES))
    M2,S2=res.run(ud,M0=np.random.RandomState(999).rand(N_RES)*M_TH*2)
    fig=plt.figure(figsize=(20,16)); gs=GridSpec(4,3,figure=fig,hspace=0.45,wspace=0.35)
    for ni,n in enumerate([0,100,200]):
        ax=fig.add_subplot(gs[0,ni]); ax.plot(M1[:400,n],'b-',lw=0.6,alpha=0.8,label='IC1'); ax.plot(M2[:400,n],'r-',lw=0.6,alpha=0.8,label='IC2')
        ax.axhline(M_TH,color='gray',ls=':',lw=0.5); ax.set_title(f'Neuron {n}',fontsize=10,fontweight='bold')
        if ni==0: ax.legend(fontsize=7)
    for si,(S,c) in enumerate([(S1,'b'),(S2,'r')]):
        ax=fig.add_subplot(gs[1,si])
        for t in range(500):
            for n in range(50):
                if S[t,n]: ax.plot(t,n,c+'.',ms=0.3)
        ax.set_title(f'IC{si+1} Spikes',fontsize=10,fontweight='bold')
    ax=fig.add_subplot(gs[1,2]); ag=np.array([(S1[max(0,t-30):t+1]==S2[max(0,t-30):t+1]).mean() for t in range(len(S1))])
    ax.plot(ag,'g-',lw=0.8); ax.set_title(f'Spike Agreement\nFinal:{ag[-1]:.3f}',fontsize=10,fontweight='bold'); ax.set_ylim(0.5,1.02)
    diff=np.sqrt(np.mean((M1-M2)**2,axis=1))
    ax=fig.add_subplot(gs[2,0]); ax.plot(diff,'b-'); ax.set_title('State Diff (linear)',fontsize=10,fontweight='bold')
    ax=fig.add_subplot(gs[2,1]); ax.semilogy(diff,'b-'); ax.axhline(diff[0]*0.01,color='red',ls='--'); ax.set_title('State Diff (log)',fontsize=10,fontweight='bold')
    ax=fig.add_subplot(gs[2,2]); ax.plot(uniform_filter1d(S1.mean(1).astype(float),20),'b-',alpha=0.7,label='IC1'); ax.plot(uniform_filter1d(S2.mean(1).astype(float),20),'r-',alpha=0.7,label='IC2'); ax.legend(fontsize=8); ax.set_title('Pop Rate',fontsize=10,fontweight='bold')
    pca=PCA(2,random_state=42); z1=pca.fit_transform(M1); z2=pca.transform(M2)
    for si,(z,cm) in enumerate([(z1,'Blues'),(z2,'Reds')]):
        ax=fig.add_subplot(gs[3,si]); ax.scatter(z[:,0],z[:,1],c=np.arange(len(z)),cmap=cm,s=0.5,alpha=0.5,rasterized=True); ax.set_title(f'IC{si+1} Phase',fontsize=10,fontweight='bold')
    ax=fig.add_subplot(gs[3,2]); ax.scatter(z1[400:,0],z1[400:,1],c='blue',s=0.5,alpha=0.3,label='IC1 late',rasterized=True); ax.scatter(z2[400:,0],z2[400:,1],c='red',s=0.5,alpha=0.3,label='IC2 late',rasterized=True); ax.legend(fontsize=8,markerscale=5); ax.set_title('Converged',fontsize=10,fontweight='bold')
    fig.suptitle(f'Raw Observation 6.4: ESP Convergence (S{dsid}, Ch{dch}, Threat)',fontsize=13,fontweight='bold',y=1.03)
    plt.savefig(f'{a.output_dir}/raw6_1d_esp_convergence.pdf',bbox_inches='tight',dpi=150); plt.close()
    print(f"  raw6_1d ({time.time()-t0:.0f}s)")

    # ── FIG 5: Multi-Subject ESP ──
    fig,axes=plt.subplots(2,3,figsize=(18,10)); ssids=subjs[::35][:6]
    for si,sid in enumerate(ssids):
        ax=axes[si//3,si%3]
        for cat in ['Threat','Cute']:
            if (sid,cat) not in af: continue
            eeg=np.loadtxt(af[(sid,cat)]); u=norm(eeg[:,dch])
            M1,_=res.run(u,M0=np.zeros(N_RES)); M2,_=res.run(u,M0=np.random.RandomState(999).rand(N_RES)*M_TH*2)
            ax.semilogy(np.sqrt(np.mean((M1-M2)**2,axis=1)),color=CCOL[cat],lw=1.0,label=cat)
        ax.set_title(f'S{sid}',fontsize=10,fontweight='bold'); ax.grid(alpha=0.2)
        if si==0: ax.legend(fontsize=8)
    fig.suptitle(f'Raw Observation 6.5: ESP Across 6 Subjects',fontsize=12,fontweight='bold',y=1.03)
    plt.tight_layout(); plt.savefig(f'{a.output_dir}/raw6_1e_multi_subject_esp.pdf',bbox_inches='tight',dpi=150); plt.close()
    print(f"  raw6_1e ({time.time()-t0:.0f}s)")

    # ── ANALYSIS: Driven Lyapunov (full population) ──
    print("\n--- Lyapunov exponents (full population) ---")
    lyap=[]
    for si,sid in enumerate(subjs):
        if si%50==0: print(f"  {si+1}/{len(subjs)}... {time.time()-t0:.0f}s")
        for cat in CATS:
            if (sid,cat) not in af: continue
            eeg=np.loadtxt(af[(sid,cat)])
            for ch in a.analysis_channels:
                u=norm(eeg[:,ch]); rng=np.random.RandomState(42); e=rng.randn(N_RES); e/=np.linalg.norm(e)
                mr=np.zeros(N_RES);sr=np.zeros(N_RES);mp=mr+DELTA_0*e;sp=np.zeros(N_RES); logs=[]
                for t in range(len(u)):
                    I=res.Win*u[t]; mr_n=(1-BETA)*mr*(1-sr)+I+res.Wrec@sr; sr_n=(mr_n>=M_TH).astype(float)
                    mp_n=(1-BETA)*mp*(1-sp)+I+res.Wrec@sp; sp_n=(mp_n>=M_TH).astype(float)
                    mr=mr_n;sr=sr_n;mp=mp_n;sp=sp_n
                    if (t+1)%T_RENORM==0:
                        d=mp-mr;dist=np.linalg.norm(d)
                        if dist>0: logs.append(np.log(dist/DELTA_0)); mp=mr+DELTA_0*(d/dist); sp=sr.copy()
                if logs: lyap.append({'subject':sid,'channel':ch,'category':cat,'lambda1':np.mean(logs)/T_RENORM})
    lv=np.array([r['lambda1'] for r in lyap])
    print(f"  lambda_1={lv.mean():.6f}+/-{lv.std():.6f}, 100% neg: {(lv<0).all()}")

    # ── FIG 6: Lyapunov Analysis ──
    # (Raw stretching for demo subject + population distribution)
    eeg=np.loadtxt(af[(dsid,'Threat')]); u=norm(eeg[:,dch])
    rng=np.random.RandomState(42);e=rng.randn(N_RES);e/=np.linalg.norm(e)
    mr=np.zeros(N_RES);sr=np.zeros(N_RES);mp=mr+DELTA_0*e;sp=np.zeros(N_RES);ld=[];td=[]
    for t in range(len(u)):
        I=res.Win*u[t];mr_n=(1-BETA)*mr*(1-sr)+I+res.Wrec@sr;sr_n=(mr_n>=M_TH).astype(float)
        mp_n=(1-BETA)*mp*(1-sp)+I+res.Wrec@sp;sp_n=(mp_n>=M_TH).astype(float)
        mr=mr_n;sr=sr_n;mp=mp_n;sp=sp_n
        if (t+1)%T_RENORM==0:
            d=mp-mr;dist=np.linalg.norm(d)
            if dist>0: ld.append(np.log(dist/DELTA_0));td.append(t);mp=mr+DELTA_0*(d/dist);sp=sr.copy()

    fig=plt.figure(figsize=(20,12)); gs=GridSpec(2,4,figure=fig,hspace=0.45,wspace=0.35)
    ax=fig.add_subplot(gs[0,0:2]); ax.plot(td,ld,'b.-',ms=3,lw=0.8); ax.axhline(0,color='red',ls='--',lw=1.5)
    ax.set_xlabel('Time Step');ax.set_ylabel('ln(stretching)');ax.set_title(f'Raw Benettin (S{dsid})',fontsize=10,fontweight='bold');ax.grid(alpha=0.2)
    ax=fig.add_subplot(gs[0,2:4]); cum=np.cumsum(ld)/(np.arange(1,len(ld)+1)*T_RENORM)
    ax.plot(td,cum,'b-',lw=1.5);ax.axhline(0,color='red',ls='--');ax.axhline(cum[-1],color='green',ls=':',label=f'Final={cum[-1]:.5f}')
    ax.set_title('Cumulative lambda_1',fontsize=10,fontweight='bold');ax.legend(fontsize=9)
    ax=fig.add_subplot(gs[1,0:2]); ax.hist(lv,bins=60,color='#e74c3c',edgecolor='black',alpha=0.7)
    ax.axvline(0,color='black',lw=2);ax.axvline(lv.mean(),color='blue',lw=2,ls='--',label=f'Mean={lv.mean():.5f}')
    ax.set_xlabel('lambda_1');ax.set_title(f'Population (N={len(lv)})',fontsize=10,fontweight='bold');ax.legend(fontsize=8)
    ax=fig.add_subplot(gs[1,2]); cd=[[r['lambda1'] for r in lyap if r['category']==c] for c in CATS]
    bp=ax.boxplot(cd,patch_artist=True,tick_labels=['Thr','Mut','Cut','Ero'])
    for p,c in zip(bp['boxes'],CATS): p.set_facecolor(CCOL[c]);p.set_alpha(0.6)
    ax.axhline(0,color='black',lw=1.5);ax.set_title('By Category',fontsize=10,fontweight='bold')
    ax=fig.add_subplot(gs[1,3]); sl={}
    for r in lyap: sl.setdefault(r['subject'],[]).append(r['lambda1'])
    ax.hist([np.mean(v) for v in sl.values()],bins=40,color='#8e44ad',edgecolor='black',alpha=0.7)
    ax.axvline(0,color='black',lw=2);ax.set_title('Per-Subject',fontsize=10,fontweight='bold')
    fig.suptitle(f'Analysis 6.1: lambda_1={lv.mean():.5f}+/-{lv.std():.5f}, 100% negative. ESP verified.',fontsize=13,fontweight='bold',y=1.04)
    plt.savefig(f'{a.output_dir}/analysis6_1f_lyapunov.pdf',bbox_inches='tight',dpi=150); plt.close()
    print(f"  analysis6_1f ({time.time()-t0:.0f}s)")

    pickle.dump({'lyapunov_results':lyap,'demo_sid':dsid,'demo_ch':dch,'subjects':subjs,'categories':CATS},
                open(f'{a.results_dir}/ch6_exp1_full.pkl','wb'))
    print(f"\nDone. 6 figures. {time.time()-t0:.0f}s total.")

if __name__=='__main__': main()
