#!/usr/bin/env python3
"""
Author every equation, table, and diagram for the ARSPI-Net defense in LaTeX/TikZ
and render each to a tight, transparent PNG (via pdflatex -> PyMuPDF).

Each item's .tex source is kept in defense_build/latex/ so it can be re-edited;
the rendered PNG lands in defense_build/figpng/.

Run:  python3 defense_build/latexgen.py
"""
import os, subprocess, sys
import fitz  # PyMuPDF

HERE = os.path.dirname(os.path.abspath(__file__))
LATEX_DIR = os.path.join(HERE, "latex")
OUT_DIR = os.path.join(HERE, "figpng")
BUILD = os.path.join(LATEX_DIR, "_build")
os.makedirs(LATEX_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(BUILD, exist_ok=True)

DPI = 360

PREAMBLE = r"""
\documentclass[border=10pt]{standalone}
\usepackage[table]{xcolor}
\usepackage{amsmath,amssymb,amsfonts,mathtools,bm}
\usepackage{booktabs,array}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,backgrounds,fit,shapes.geometric,decorations.pathreplacing}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{sbnavy}{HTML}{1A2A4F}
\definecolor{sbteal}{HTML}{1C7293}
\definecolor{sbaccent}{HTML}{B5651D}
\definecolor{sbgrey}{HTML}{8A94A6}
\definecolor{sblight}{HTML}{EAF1F5}
\newcommand{\cmark}{\textcolor{sbteal}{\ensuremath{\boldsymbol{\checkmark}}}}
\newcommand{\xmark}{\textcolor{sbgrey}{\ensuremath{\boldsymbol{\times}}}}
\newcommand{\eqcol}{\color{sbnavy}}
"""

def MATH(body, size=26):
    return PREAMBLE + r"\begin{document}" + \
        r"{\fontsize{%d}{%d}\selectfont \eqcol $\displaystyle %s$}" % (size, int(size*1.25), body) + \
        r"\end{document}"

def RAW(body):
    return PREAMBLE + r"\begin{document}" + body + r"\end{document}"

# ----------------------------------------------------------------------------- EQUATIONS
EQUATIONS = {
"eq_observation": r"x_p(s) \;=\; g\!\big(r_p(s)\big) \;+\; \eta",
"eq_inverse":     r"\phi:\quad x_p \;\longmapsto\; \widehat{D}_p",
"eq_lif":         r"\tau_m\,\frac{dV_i}{dt} \;=\; -\big(V_i - V_{\mathrm{rest}}\big) + R\,I_i(t)\,,\qquad \beta = e^{-\Delta t/\tau_m}",
"eq_bsc6":        r"\mathbf{B}_{c,b} \;=\; \sum_{t\,\in\,W_b}\mathbf{S}^{(c)}(t)\,,\qquad b = 1,\dots,6",
"eq_chain":       r"x_c(t)\ \xrightarrow{\ \text{LIF}\ }\ \mathbf{S}_c(t)\ \xrightarrow{\ \text{BSC}_6\ }\ \mathbf{B}_c\in\mathbb{R}^{1536}\ \xrightarrow{\ \text{PCA-64}\ }\ \mathbf{H}_c\in\mathbb{R}^{64}",
"eq_centering":   r"\widetilde{\mathbf{H}}_p \;=\; \mathbf{H}_p - \bar{\mathbf{H}}_p",
"eq_rho":         r"\rho \;=\; \frac{\sigma^2_{\text{subject}}}{\sigma^2_{\text{condition}}} \;=\; 7.2",
"eq_lyap":        r"\lambda_1^{(\text{driven})} < 0 \qquad\Longrightarrow\qquad \lambda_1 = -0.054",
"eq_prop":        r"\mathbf{H}^{(k+1)} = \sigma\!\big(\mathbf{W}^{(k)}\,\tilde{\mathbf{A}}\,\mathbf{H}^{(k)}\big),\qquad \tilde{\mathbf{A}} = \mathbf{D}^{-1/2}(\mathbf{A}+\mathbf{I})\,\mathbf{D}^{-1/2}",
"eq_kappa":       r"\kappa_{s,c} \;=\; \frac{\lVert \mathbf{C}_{s,c}\rVert_F}{\sqrt{p\,q}} \;=\; \frac{1}{\sqrt{14}}\sqrt{\sum_{j=1}^{7}\sum_{k=1}^{2}\big(\mathbf{C}_{s,c}\big)_{jk}^{2}}",
"eq_koopman":     r"(\mathcal{K}\,g)(\mathbf{x}) \;=\; g\!\big(\mathbf{F}(\mathbf{x})\big)",
}

# ----------------------------------------------------------------------------- TABLES
TABLES = {
"tbl_landscape": r"""
{\sffamily\fontsize{20}{26}\selectfont
\setlength{\arrayrulewidth}{1pt}\renewcommand{\arraystretch}{1.45}
\begin{tabular}{l c c c c}
\toprule
\rowcolor{sbnavy}\textbf{\textcolor{white}{Interpretability level}} & \textbf{\textcolor{white}{ARSPI-Net}} & \textbf{\textcolor{white}{NeuCube}} & \textbf{\textcolor{white}{ProtoPMed}} & \textbf{\textcolor{white}{EEGNet+SHAP}}\\
\midrule
L1\;Temporal traceability      & \cmark & \cmark & \xmark & $\sim$\\
\rowcolor{sblight}L2\;Geometric transparency     & \cmark & \xmark & \cmark & $\sim$\\
L3\;Dynamical characterization & \cmark & \cmark & \xmark & \xmark\\
\rowcolor{sblight}L4\;Systems coupling           & \cmark & \xmark & \xmark & \xmark\\
\midrule
\textbf{Intrinsic levels} & \textcolor{sbaccent}{\textbf{4}} & 2 & 1 & 0\\
\bottomrule
\end{tabular}}
""",
"tbl_baseline": r"""
{\sffamily\fontsize{20}{26}\selectfont
\setlength{\arrayrulewidth}{1pt}\renewcommand{\arraystretch}{1.4}
\begin{tabular}{l c c c}
\toprule
\rowcolor{sbnavy}\textbf{\textcolor{white}{Model}} & \textbf{\textcolor{white}{Uncentered}} & \textbf{\textcolor{white}{Centered}} & \textbf{\textcolor{white}{$\Delta$}}\\
\midrule
EEGNet              & 72.0 & 89.1 & $+17.1$\\
Raw EEG $+$ linear  & 70.5 & 88.4 & $+17.9$\\
PCA-200             & 64.9 & 86.4 & $+21.5$\\
GRU                 & 59.9 & 78.4 & $+18.5$\\
\rowcolor{sbaccent!22}\textbf{ARSPI-Net} & \textbf{59.4} & \textbf{78.8} & $\mathbf{+19.4}$\\
LSTM                & 58.0 & 71.1 & $+13.1$\\
Band power $+$ SVM  & 47.7 & 61.0 & $+13.3$\\
\bottomrule
\end{tabular}}
""",
"tbl_taxonomy": r"""
{\sffamily\fontsize{20}{26}\selectfont
\setlength{\arrayrulewidth}{1pt}\renewcommand{\arraystretch}{1.5}
\begin{tabular}{l l l}
\toprule
\rowcolor{sbnavy}\textbf{\textcolor{white}{Level}} & \textbf{\textcolor{white}{Observable}} & \textbf{\textcolor{white}{Measured}}\\
\midrule
L1\;Temporal  & bin vs.\ ERP scalar             & $|r| = 0.82$ \;(LPP)\\
\rowcolor{sblight}L2\;Geometric & subject vs.\ condition variance & $\rho = 7.2$;\ \ $+13$--$21$\,pp\\
L3\;Dynamical & descriptor vs.\ ERP scalar      & $|r|$ up to $0.84$\\
\rowcolor{sblight}L4\;Coupling  & $\kappa$ vs.\ permutation null  & $0.273$,\ \ $p<0.001$\\
\bottomrule
\end{tabular}}
""",
"tbl_disorder": r"""
{\sffamily\fontsize{20}{26}\selectfont
\setlength{\arrayrulewidth}{1pt}\renewcommand{\arraystretch}{1.5}
\begin{tabular}{l l c}
\toprule
\rowcolor{sbnavy}\textbf{\textcolor{white}{Disorder}} & \textbf{\textcolor{white}{Most-sensitive layer}} & \textbf{\textcolor{white}{$p$}}\\
\midrule
\rowcolor{sbaccent!22}SUD  & Dynamical descriptors & $\mathbf{0.0004}$\\
PTSD & Dynamical descriptors & $0.036$\\
GAD  & Graph topology        & $0.032$\\
ADHD & Cross-layer coupling  & ---\\
\bottomrule
\end{tabular}}
""",
"tbl_generations": r"""
{\sffamily\fontsize{20}{26}\selectfont
\setlength{\arrayrulewidth}{1pt}\renewcommand{\arraystretch}{1.5}
\begin{tabular}{l l l}
\toprule
\rowcolor{sbnavy}\textbf{\textcolor{white}{Generation}} & \textbf{\textcolor{white}{Mechanism}} & \textbf{\textcolor{white}{Time}}\\
\midrule
1st \;\textemdash\; McCulloch--Pitts & threshold logic              & none\\
\rowcolor{sblight}2nd \;\textemdash\; deep learning      & dense, synchronous rates     & implicit\\
3rd \;\textemdash\; spiking (Maass)  & sparse asynchronous events   & \textcolor{sbaccent}{\textbf{explicit}}\\
\bottomrule
\end{tabular}}
""",
}

# ----------------------------------------------------------------------------- TIKZ DIAGRAMS
TIKZ = {
# Volume conduction: layered head + dipole smearing to electrodes
"tikz_head": r"""
\begin{tikzpicture}[font=\sffamily]
  \foreach \r/\col/\lab/\sig in {3.4/sbnavy!12/scalp/{\footnotesize scalp}, 2.85/sbgrey!35/skull/{\footnotesize skull (insulator)}, 2.4/sbteal!22/CSF/{\footnotesize CSF}, 2.0/sbaccent!20/cortex/{\footnotesize cortex}}{
     \fill[\col] (0,0) circle (\r);
  }
  \draw[sbnavy,line width=1pt] (0,0) circle (3.4);
  \draw[sbgrey,line width=0.8pt] (0,0) circle (2.85);
  \draw[sbteal,line width=0.8pt] (0,0) circle (2.4);
  \draw[sbaccent,line width=0.8pt] (0,0) circle (2.0);
  % dipole source
  \draw[-{Stealth[length=4mm]},sbaccent,line width=2pt] (0,1.0) -- (0,1.7);
  \node[sbaccent,font=\footnotesize\bfseries] at (1.15,1.35) {dipole};
  % electrodes on scalp (top arc)
  \foreach \a in {50,70,90,110,130}{
     \fill[sbnavy] (\a:3.4) circle (2.2pt);
  }
  % smeared spread (dashed) from dipole to several electrodes
  \foreach \a in {50,70,90,110,130}{
     \draw[sbteal!70,dashed,line width=0.7pt] (0,1.35) to[bend left=10] (\a:3.34);
  }
  % labels at right
  \node[anchor=west,font=\footnotesize] at (3.7,2.0)  {scalp \;(conductive)};
  \node[anchor=west,font=\footnotesize] at (3.7,1.2)  {skull \;(high resistance)};
  \node[anchor=west,font=\footnotesize] at (3.7,0.4)  {CSF \;(conductive)};
  \node[anchor=west,font=\footnotesize] at (3.7,-0.4) {cortex \;(source)};
  \node[anchor=west,font=\footnotesize\itshape,text=sbnavy] at (3.7,-1.4) {one source $\rightarrow$ many sensors};
  \node[anchor=west,font=\footnotesize\itshape,text=sbnavy] at (3.7,-2.0) {(spatial low-pass filter)};
\end{tikzpicture}
""",
# EEG vs fMRI temporal/spatial resolution map
"tikz_resolution": r"""
\begin{tikzpicture}[font=\sffamily]
  \draw[-{Stealth[length=3mm]},line width=1pt] (0,0) -- (9.2,0) node[below right,font=\small] {temporal resolution};
  \draw[-{Stealth[length=3mm]},line width=1pt] (0,0) -- (0,6.0) node[above left,font=\small,align=center] {spatial\\ resolution};
  % axis ticks (log-ish, conceptual)
  \foreach \x/\t in {1/ms,3/10\,ms,5/0.1\,s,7/1\,s,8.6/10\,s}{ \node[below,font=\scriptsize] at (\x,0) {\t};}
  \foreach \y/\t in {1.2/cm,3/mm,4.8/sub-mm}{ \node[left,font=\scriptsize] at (0,\y) {\t};}
  % fast axis annotation
  \node[font=\scriptsize,sbgrey] at (4.6,-0.75) {faster $\rightarrow$};
  % EEG region: fast temporal, coarse spatial (upper-left)
  \fill[sbaccent!25] (1.6,1.1) ellipse (1.25 and 0.85);
  \draw[sbaccent,line width=1.2pt] (1.6,1.1) ellipse (1.25 and 0.85);
  \node[sbaccent,font=\bfseries] at (1.6,1.1) {EEG};
  \node[font=\scriptsize,text=sbaccent,align=center] at (1.6,2.25) {millisecond timing\\ ``the music''};
  % fMRI region: slow temporal, fine spatial (lower-right)
  \fill[sbteal!22] (7.1,4.4) ellipse (1.25 and 0.9);
  \draw[sbteal,line width=1.2pt] (7.1,4.4) ellipse (1.25 and 0.9);
  \node[sbteal,font=\bfseries] at (7.1,4.4) {fMRI};
  \node[font=\scriptsize,text=sbteal,align=center] at (7.1,5.7) {seconds-lagged BOLD};
\end{tikzpicture}
""",
# Reservoir / liquid state machine schematic
"tikz_reservoir": r"""
\begin{tikzpicture}[font=\sffamily,>=Stealth]
  \node[align=center,font=\small] (in) at (0,0) {\textbf{input}\\[2pt]{\scriptsize spike-encoded EEG}\\[2pt]$u(t)$};
  \node[draw=sbnavy,line width=1.2pt,rounded corners,fill=sbnavy!7,minimum width=4.2cm,minimum height=4.0cm] (res) at (4.6,0) {};
  \node[above=1mm of res,sbnavy,font=\bfseries] {fixed reservoir};
  \foreach \p in {(3.4,1.2),(4.2,1.5),(5.2,1.1),(5.8,1.5),(3.6,0.2),(4.6,0.4),(5.5,0.1),(3.4,-0.9),(4.3,-1.2),(5.3,-0.9),(5.9,-0.2),(4.0,-0.4)}{
     \fill[sbteal] \p circle (3.2pt);}
  \draw[sbgrey,->] (3.4,1.2)--(4.3,-1.2);
  \draw[sbgrey,->] (4.2,1.5)--(5.5,0.1);
  \draw[sbgrey,->] (4.6,0.4)--(5.9,-0.2);
  \draw[sbgrey,->] (3.6,0.2)--(5.3,-0.9);
  \draw[sbgrey,->] (4.0,-0.4)--(4.2,1.5);
  \draw[sbgrey,->] (5.2,1.1)--(3.4,-0.9);
  \node[below=1mm of res,align=center,font=\scriptsize,sbnavy] {\textbf{weights random \& fixed} --- never trained};
  \node[draw=sbaccent,line width=1pt,rounded corners,fill=sbaccent!12,minimum width=2cm] (r1) at (9.0,1.0) {readout};
  \node[draw=sbaccent,line width=1pt,rounded corners,fill=sbaccent!12,minimum width=2cm] (r2) at (9.0,-0.2) {readout};
  \node[font=\small] at (9.0,-1.0) {$\vdots$};
  \draw[->,line width=1pt] (in.east) -- (res.west);
  \draw[->,line width=1pt] (res.east) -- (r1.west);
  \draw[->,line width=1pt] (res.east) -- (r2.west);
  \node[below=5mm of r2,align=center,font=\scriptsize,sbaccent] {only the linear\\ readout is trained};
\end{tikzpicture}
""",
# Cover's theorem: lift to separability
"tikz_cover": r"""
\begin{tikzpicture}[font=\sffamily]
  % left: 2D not separable (two interleaved rings)
  \begin{scope}
    \draw[sbgrey,line width=0.8pt] (-0.2,-0.2) rectangle (3.4,3.2);
    \foreach \a in {0,40,...,320}{ \fill[sbteal] ($(1.6,1.5)+(\a:0.6)$) circle (2pt);}
    \foreach \a in {20,60,...,340}{ \fill[sbaccent] ($(1.6,1.5)+(\a:1.35)$) circle (2pt);}
    \node[font=\small,sbnavy] at (1.6,-0.6) {input space: not separable};
  \end{scope}
  \draw[-{Stealth[length=4mm]},line width=1.6pt,sbnavy] (3.8,1.5) -- (5.2,1.5) node[midway,above,font=\small]{$\phi$};
  % right: lifted, separable by a plane
  \begin{scope}[shift={(6.0,0)}]
    \draw[sbgrey,line width=0.8pt] (-0.2,-0.2) rectangle (3.7,3.2);
    \fill[sbteal] (0.7,0.5) circle (2pt) (1.2,0.7) circle (2pt) (1.8,0.4) circle (2pt) (2.4,0.7) circle (2pt) (3.0,0.5) circle (2pt) (1.0,0.3) circle (2pt) (2.0,0.6) circle (2pt);
    \fill[sbaccent] (0.6,2.5) circle (2pt) (1.3,2.7) circle (2pt) (1.9,2.4) circle (2pt) (2.6,2.7) circle (2pt) (3.1,2.5) circle (2pt) (1.0,2.6) circle (2pt) (2.2,2.5) circle (2pt);
    \draw[sbnavy,dashed,line width=1.2pt] (-0.1,1.55) -- (3.6,1.55);
    \node[font=\small,sbnavy] at (1.75,-0.6) {high-D: linearly separable};
  \end{scope}
\end{tikzpicture}
""",
# Neuromorphic vs CMOS energy per operation (accurate to cited numbers)
"tikz_energy": r"""
\begin{tikzpicture}
\begin{axis}[
  font=\sffamily, width=10.5cm, height=6cm, ybar, bar width=52pt,
  ymode=log, ymin=10, ymax=2500, ylabel={energy / operation (pJ, log scale)},
  symbolic x coords={CMOS MAC, Neuromorphic}, xtick=data,
  nodes near coords style={font=\bfseries, anchor=south},
  axis lines=left, ymajorgrids, grid style={sbgrey!30},
  enlarge x limits=0.6, tick label style={font=\small}]
  \addplot[draw=sbnavy,fill=sbnavy!75,nodes near coords={900~pJ}] coordinates {(CMOS MAC,900)};
  \addplot[draw=sbaccent,fill=sbaccent!80,nodes near coords={20~pJ}] coordinates {(Neuromorphic,20)};
\end{axis}
\node[font=\sffamily\bfseries,sbaccent,align=center] at (5.4,3.1) {$\approx 45\times$\\ lower};
\end{tikzpicture}
""",
}

def render(name, tex, repair_passes=1):
    texpath = os.path.join(LATEX_DIR, name + ".tex")
    with open(texpath, "w") as f:
        f.write(tex)
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
           "-output-directory", BUILD, texpath]
    ok = True
    for _ in range(repair_passes):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            ok = False
            break
        ok = True
    pdfpath = os.path.join(BUILD, name + ".pdf")
    if not ok or not os.path.exists(pdfpath):
        logpath = os.path.join(BUILD, name + ".log")
        tail = ""
        if os.path.exists(logpath):
            tail = "\n".join(open(logpath, errors="ignore").read().splitlines()[-22:])
        print(f"FAIL  {name}\n{tail}\n")
        return False
    doc = fitz.open(pdfpath)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(DPI/72, DPI/72), alpha=True)
    pix.save(os.path.join(OUT_DIR, name + ".png"))
    print(f"OK    {name}  ({pix.width}x{pix.height})")
    doc.close()
    return True

def main():
    items = []
    for n, b in EQUATIONS.items(): items.append((n, MATH(b)))
    for n, b in TABLES.items():    items.append((n, RAW(b)))
    for n, b in TIKZ.items():      items.append((n, RAW(b)))
    fails = []
    for n, tex in items:
        if not render(n, tex):
            fails.append(n)
    print(f"\n=== rendered {len(items)-len(fails)}/{len(items)} ; failures: {fails} ===")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
