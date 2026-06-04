# ARSPI-Net — Doctoral Defense: Spoken Transcript

*Andrew Lane · Stony Brook ECE · ~40 minutes. Word-for-word script; the same text is in each slide's speaker notes.*

---

### Title

Good morning, and thank you all for being here. I'm not going to open with an architecture. I'm going to open with a decision a doctor has to make, because that decision is the reason this whole project exists.

### Slide 1 — An AI reads the scan. It's 89% sure of the diagnosis.
It even names the drug. Then you ask why — and it has nothing.

Picture a psychiatrist with a patient who has severe, treatment-resistant depression. An AI tool reads the patient's EEG and returns a diagnosis, eighty-nine percent confident, and it even recommends a specific drug. That sounds like the future of medicine — until the doctor asks the one question that matters. Why? What in this brain led to that number? And the model has nothing to say, because it doesn't know. It only knows the pattern matched.

### Slide 2 — The same machine, the same confidence

Here's what I mean. This is the same kind of model that looks at this photo and says American goldfinch, eighty-seven percent — and it's right. But show it something that plainly isn't a bird, and it is just as confident. From the outside, right for the right reasons and right for the wrong reasons look identical. In a clinic, that isn't a curiosity. It's a liability.

### Slide 3 — Can we recover the brain's hidden dynamics from a
noisy, low-dimensional trace — and keep an explanation
a clinician can actually read?

So here is the question I set out to answer. Can we recover the hidden dynamics of the brain from a noisy, low-dimensional trace — and do it with a model whose internal workings a clinician can actually read? Both halves matter. Accuracy alone isn't enough, and an explanation of a wrong mechanism is worse than none. Everything from here is how I built a system that answers yes to both.


## ◆ The measurement problem

*So let me start with the signal itself — where it comes from, and why a clean-looking trace is so deceptively hard to read.*

### Slide 4 — Recovering a hidden system from one trace

Recovering a hidden system from a thin signal is not a fantasy — physics does it all the time. NASA put a single seismometer on the surface of Mars, and from that one squiggle reconstructed the planet's deep interior, structure no instrument could ever touch directly. The trick is that they knew the physics connecting the inside to the surface. That is exactly the bet I'm making with the brain.

### Slide 5 — Where the EEG signal comes from

An EEG signal is born when thousands of cortical pyramidal cells fire together and act like a tiny electrical dipole. What we actually record on the scalp isn't that source — it's a noisy, smeared, averaged projection of it, like these evoked waveforms. Formally: the measurement x is some mixing function g of the latent cortical response r, plus noise. The whole game is getting back to r.

### Slide 6 — Volume conduction makes it an inverse problem

The reason this is hard is volume conduction. The signal has to cross cerebrospinal fluid, the skull — which is a terrible conductor — and the scalp. Each layer smears it. One source spreads across many electrodes, and each electrode sees a blur of the whole cortex. So I'm inverting that: given the scalp trace, recover the latent state. With thirty-four channels standing in for billions of neurons, that map is massively underdetermined. It's shadows on a cave wall — and the rest of the talk is how to read them anyway.

### Slide 7 — Why EEG and not fMRI

People ask why not just use fMRI, with its beautiful spatial maps. Because fMRI measures blood flow — a sluggish, metabolic echo that lags the actual neural event by seconds. Emotion and threat detection happen in milliseconds. Using fMRI for that is like following a fast piano piece by measuring the temperature of the keys: you learn which were pressed, but the music is gone. EEG keeps the timing. It gives us the music — and the discriminative information lives in that timing.

### Slide 8 — The data: real patients, not textbook cases

And the data are real. The SHAPE cohort is two hundred eleven adults, and it's transdiagnostic — these aren't clean textbook cases. The average patient carries about three overlapping diagnoses: depression, PTSD, substance use, anxiety. We've known for decades that these conditions show up in EEG at the group level. But you only get that clean group signal by averaging away the individual — and you can't average away the person sitting in front of you. That gap is the whole clinical problem.


## ◆ Why a black box is the wrong tool here

*That sets up the real obstacle. It isn't getting an answer out of the data — modern networks do that easily. It's getting an answer you can trust.*

### Slide 9 — Post-hoc explanations explain the shortcut, not the science

The standard rebuttal is that we've solved the black box — just run SHAP or a saliency map. But those explain the model from the outside; they approximate where it looked, not why. If a network secretly learned to key off the hum of the EEG machine in room four, a saliency map will faithfully, confidently highlight that artifact. It's the Clever Hans effect with a colorful heatmap on top. It can make a broken model look trustworthy, which in medicine is the most dangerous outcome of all.

### Slide 10 — Intrinsic interpretability instead

I want the opposite: a model whose internal variables are the explanation. Here's the contrast on the same data. The black box's attention peaks late, between four hundred and seven hundred milliseconds — outside the windows that decades of clinical work tie to emotional processing. It's accurate, but for reasons a neurologist can't use. ARSPI-Net's representation sits right on the named event-related components — the P300, the early late-positive potential. One is a gradient about a model. The other is an account in the brain's own units.

### Slide 11 — The claim is the count, not the accuracy

So here's where this work sits. A black box with SHAP gives you zero intrinsic levels of interpretability. Prototype methods give one. NeuCube, the closest neighbor, gives two. ARSPI-Net exposes four, in a single pipeline. That count — not a leaderboard number — is the contribution, and I'll show you all four levels measured.


## ◆ The approach: a spiking reservoir

*So here's the machine I built to thread that needle — and why every piece of it is chosen from theory, not tuned for a score.*

### Slide 12 — Spikes are events in time

The brain doesn't compute in clocked, dense arithmetic. It computes in spikes — sparse events whose timing carries the information. That's the third generation of neural models, and it's a natural match for a signal whose content is in its timing. My unit is the leaky integrate-and-fire neuron: it integrates input until it crosses threshold, fires, and resets. One parameter, the leak beta, sets how long it remembers.

### Slide 13 — Spikes are also almost free

There's a second reason to care about spikes, beyond the physics — energy. A dense multiply-accumulate on conventional hardware costs around nine hundred picojoules; on event-driven neuromorphic silicon the same operation is roughly twenty. That's about forty-five times less, and it's what could eventually take this off a server and onto a wearable at the bedside. Everything here ran in software, so I treat that number as motivation and a future on-chip measurement — not a result I'm claiming today.

### Slide 14 — The reservoir: a calibrated ruler, not an oracle

I wire two hundred fifty-six of these neurons into a recurrent pool — a reservoir. Here's the intuition: throw a rock into a still pond. You can't measure the rock in flight, but the ripples it leaves are a faithful record of its size and speed. The input is the rock, the reservoir is the water, and a simple readout reads the ripples. Now the decision that makes everything else in this talk possible — the recurrent weights are random, and they are fixed. I never train them. The instant you train the water, it starts bending its own physics to chase a score, and the ripples stop being an honest record of the rock. A fixed reservoir stays a calibrated instrument. And because nothing inside it is trained, its entire state is a pure, reproducible function of the input — which is exactly what lets me read it later, neuron by neuron.

### Slide 15 — Why this is recoverable, not a hope

And this is provably the right move, not a lucky one — which matters, because a committee will ask why this should work at all. Cover's theorem says that if you lift tangled data into a high enough dimensional space, it almost always becomes linearly separable; those are the ripples spreading out. Takens tells me the lift preserves the geometry of the underlying dynamics — and rather than just cite it, I measured the embedding dimension and found the trajectory needs fewer than sixty-four dimensions to live in. Koopman closes the loop: in the right space of observables, even a nonlinear system evolves linearly, which is the deep reason a plain linear readout is enough. Three results, one conclusion — a training-free core isn't a compromise. It's sufficient by construction.

### Slide 16 — ARSPI-Net, end to end

Putting it together: the scalp signal drives the fixed reservoir; the spikes get a compact temporal code; that feeds a graph stage over the electrodes; and a linear readout produces the affective state. Five stages, and under each one sits a different, measurable level of interpretability. Nothing in the core is trained — the only fitted object in the whole system is that final linear readout.

### Slide 17 — Four contributions

That gives four contributions. A neuromorphic operator I can analyze as a dynamical system. A spike code that stays tied to clinical components. A design rule that says when graph message-passing helps and when it hurts. And the one I'm most excited about — a geometric correction that lifts every model in the field, not just mine. Let me take you through the evidence.


## ◆ What the model measures

*Now the evidence. Four levels of it — and I want each one to be a number you can check, not an adjective.*

### Slide 18 — The setup, briefly

A word on discipline, because it's what makes the numbers mean something. Every split is subject-disjoint — no patient appears in both train and test. I compare representations with linear readouts, so a difference reflects the representation and not a fancier classifier. Everything is permutation-tested, and the whole pipeline is stable to the random seed within one percent. Look at this gallery for a second — every subject's signature looks completely different. Hold that thought, because it comes back as the central result.

### Slide 19 — The reservoir is stable — measured, not assumed

First, stability — and I'm careful here, because this is where people wave their hands. The usual move is to say the spectral radius is below one, so the system is stable. But that's a property of the weights sitting in a drawer; it says nothing about what a real, structured signal does once it's driving the network. So I measured the thing that actually matters: the largest Lyapunov exponent under genuine ERP drive. Across more than three thousand real trials it's minus zero point zero five four, and negative on every single one. Two nearby states, given the same input, converge — the reservoir forgets where it started. That's the echo-state property as a measurement, not an assumption, and it's what licenses everything downstream.

### Slide 20 — A system you can fully watch

And because nothing is trained, I can watch the entire internal state — all two hundred fifty-six membrane potentials as a real ERP drives the pool. You see the input-locked burst, then the graded relaxation: fading memory, as a measured fact rather than an assumption. This complete observability is the substrate for every interpretability claim that follows.

### Slide 21 — Level 1 — a temporal code clinicians can read

Now the four levels. Level one is temporal traceability. The reservoir spits out a dense spike matrix, and I funnel it through a binned spike count — six windows. Six isn't arbitrary: those windows line up with the canonical ERP components clinicians already read, the N1, P200, P300, and late-positive potential. So when the model flags an anomaly, a clinician can see which window it lives in and check it against forty years of literature.

### Slide 22 — Every step has a name

From there it's fully traceable: raw channel, spikes, the six-bin code, and a sixty-four-dimensional embedding. Every arrow is a named object, and the whole chain is stable to the seed. There's no hidden layer where the meaning disappears.

### Slide 23 — Level 2 — the discovery that reorganized the project

Level two is where the whole project turned, and I almost misread it as a failure. You'd hope the emotional condition is what drives the variance in the embedding. It accounts for under nine percent. Subject identity — your cortical folding, the thickness of your skull, your baseline rhythm — accounts for sixty-three. Identity pulls more than seven times harder than the signal I'm actually after. And worse, the two aren't in separate directions I could just project away; they share the same principal axes, so any filter that removes identity takes the emotion with it. For a while I thought the data was telling me there was nothing there. It was telling me the opposite — the signal was real, just hiding underneath a much larger one.

### Slide 24 — The fix is geometric, and it's label-free

The fix is almost embarrassingly simple. For each patient, I compute their own mean in the embedding space and subtract it — I move their origin to the center. That erases the identity offset while leaving the direction of the emotional response completely intact. It uses no labels. It's pure geometry. And seeing it was only possible because the space was transparent enough to look at.

### Slide 25 — Centering is the dominant move — for everyone

And here's the part that still gets me. The fix — re-centering each subject on their own mean — doesn't just help my model. Read down the centered column: every single model jumps by thirteen to twenty-one points. EEGNet goes from seventy-two to eighty-nine; mine from fifty-nine to seventy-nine. The emotional signal was in the data the whole time, in every representation, just buried under subject geometry. After centering, my training-free reservoir even ties a trained GRU with zero trainable recurrent parameters. A transparent model is what let me see that geometry and name the problem — and naming it raised the ceiling for the entire field, black boxes included. The geometric correction matters more than the architecture race — and that's the case for interpretability in a single number.

### Slide 26 — Why it works

Mechanically: before centering, two trials from the same person under different emotions are closer together than two trials of the same emotion from different people. The ratio is above one — identity wins. Subtract each subject's mean and it flips below one — now the conditions separate. Three-class accuracy moves from sixty-three to seventy-nine on that one geometric move.

### Slide 27 — A boundary that tells you something

One more finding I didn't expect. When I push from three classes to four, the ranking flips — the reservoir's advantage of twelve and a half points becomes a six-point deficit. That's not a bug; it's the memory-versus-nonlinearity trade-off showing me exactly where this representation pays off: when the structure lives in timing, not in amplitude. The boundary itself is information.

### Slide 28 — Level 3 — seven named observables

Level three reads the reservoir's dynamics directly, through seven named descriptors — firing rate, temporal sparsity, a memory timescale, and so on. These are physical quantities, not learned features, and that distinction matters: each one is something a physiologist can actually argue with. All seven separate emotional from neutral input at the subject level. And one of them, the autocorrelation timescale, has a direct clinical reading — an abnormally slow decay is the network failing to downregulate after a stressor, which is exactly the dynamics you'd expect from rumination. So the descriptor isn't just predictive; it's a hypothesis about mechanism.

### Slide 29 — Timing carries the signal

And when I group those descriptors, the temporal-structure family carries about two and a half times the signal of the amplitude family — independent confirmation that timing is where the information is. Better still, the seven collapse onto a single axis, from excitable to persistent. So the model doesn't just stamp a label on a patient; it locates them on a continuous, interpretable axis. That's the instrument, made concrete.

### Slide 30 — Level 4 — message passing fails here, and the theory said it would

Level four is the graph across electrodes — and I expected message passing, the field's default, to help. But that operator is mathematically a low-pass filter, and over-smoothing theory makes a sharp prediction for a small, dense montage like ours: the contrast between channels should collapse within a couple of steps. So I swept every operator I could — plain smoothing, residual connections, attention — and every one of them loses to doing no propagation at all. I went looking for a tool and found a wall. But the wall is the theory being confirmed, and it turns into a design rule the EEG-graph literature didn't actually have.

### Slide 31 — The mechanism, measured

And I can show the mechanism on the real features: as you stack graph layers, the Dirichlet energy — the contrast between channels — drops eighty-four percent by depth two, and the channels become nearly identical. That's the same depth where accuracy falls off a cliff. The diagnosis and the symptom line up.

### Slide 32 — So is the system actually one coherent thing?

So instead of forcing the graph to classify, I use it to ask a deeper question: are the local dynamics and the global topology two views of one system, or two unrelated things? I measure their coupling — kappa — against a null where I shuffle the electrode labels. The answer is that they're genuinely linked: a median of zero point two seven three, significant across all two hundred eleven subjects. It's a consistent system-wide effect rather than a strong local one, and I report it exactly that way — but it means the pieces of this pipeline are describing one coherent object, which is the last thing I'd otherwise be able to claim.

### Slide 33 — Four levels, all measured

So there are the four levels, each one a number, not an adjective. Temporal traceability correlates with the ERP at point eight two. Geometric transparency is the seven-point-two ratio and the centering lift. The dynamical descriptors track the ERP up to point eight four. And the systems coupling clears its null at high significance. To my knowledge this is the first neuromorphic EEG model to put a measurement on all four in a single pipeline.

### Slide 34 — Different disorders live in different layers

And because the model decomposes into named layers, I can ask which layer is most sensitive to which condition. Substance use separates most strongly in the dynamical descriptors, at p equals zero point zero zero zero four, and it survives correction — that's the solid clinical result. The others point in suggestive directions we're now testing with our clinical collaborators. The point isn't any single p-value; it's that a black-box score can't even pose this question. This model can.


## ◆ What it means, and where it goes

*Let me pull all of this together and say plainly what it adds up to — and what it doesn't.*

### Slide 35 — 78.8% with a full decomposition,
or 89.1% from a box you can't open.

Let me be direct about the trade at the center of this, because it's the obvious line of attack. My model gets seventy-nine percent with a complete, four-level account of why. The black box gets eighty-nine with nothing you can open. And I'll defend the seventy-nine — because in a clinic, an eighty-nine that can't justify itself is exactly the liability we opened with, while a seventy-nine you can trace end to end is an instrument a doctor can stand behind. It isn't even a large gap, and my training-free reservoir ties a fully trained recurrent network with no trained recurrence at all. Those points are the price of a glass box, and for this problem it's a price worth paying.

### Slide 36 — The biological prism

This is what I mean by a biological prism. Instead of one verdict, the model separates a patient's signal into where the abnormality lives — is it local timing, global network routing, or the coupling between them? That's a profile a clinician can reason about and act on. It's the difference between a number and an explanation.

### Slide 37 — What's next

Where this goes next — and I'll be honest about what's still open. I worked with trial-averaged data, so single-trial decoding is the natural step. The energy argument has been exactly that, an argument, so the real test is on-chip, on neuromorphic hardware like Loihi, where I can finally measure it instead of motivating it. And the layer-specific clinical findings move into proper replication with the SHAPE lab. None of these is an afterthought — the architecture was built so each one is a direct extension rather than a redesign.

### Slide 38 — The question, answered

So, back to the question I opened with. Can we recover hidden dynamics from a noisy trace and keep the explanation? The reservoir is stable — I measured it. The signal is compressible — six bins and sixty-four dimensions. The system is coherent — the coupling is real. And it's interpretable at four levels at once. With this design, the architecture isn't wrapped in an explanation. The architecture is the explanation.

### Closing

I'll close where I started — with that doctor and that patient. An instrument they can actually read changes what's possible in that room. Thank you to my advisor, Professor Tang; to Professor Nelson and the SHAPE lab; to my committee; and to my family. I'd be glad to take your questions.


## ◆ Backup — for discussion

### Slide 39 — Why a fixed reservoir, not a trained deep SNN

If you ask why I didn't train the reservoir: training a deep spiking network over hundreds of time steps needs backpropagation-through-time with surrogate gradients, and the forward and backward passes diverge over long horizons — it's brittle. A fixed reservoir sidesteps that entirely. More importantly, universality already guarantees a fixed pool plus a linear readout is enough, so I'm not giving anything up — and training the recurrence would destroy the very transparency that is the whole point.

### Slide 40 — Spectral radius vs. measured stability

If you press on stability: the autonomous spectral radius is zero point two six five, but that's the weights in isolation. What governs behavior is the driven exponent under the real input, and that's what I measured at minus zero point zero five four. I'm reporting the system in operation, not a textbook proxy.

### Slide 41 — Memory-capacity regime and the choice of leak

On why this leak and why six bins: the leak of zero point zero five sits on the memory-capacity plateau, at ninety-one percent of the peak, and it's set by matching the membrane timescale to the length of the ERP. Finer binning doesn't buy accuracy. It's a principled operating point, and a formal bin-count study is the defined next step.

### Slide 42 — Embedding dimension, measured

Takens motivates the geometry, but I don't lean on it as a guarantee — I measured the embedding dimension with false-nearest-neighbors, and it sits well below the sixty-four I project into. The reservoir has comfortable margin.

### Slide 43 — Full model comparison (N = 211)

Here's the complete two-hundred-eleven-subject comparison under one protocol. The reservoir matches the trained recurrent models with no trainable recurrence, and the ordering is exactly what the data-processing inequality predicts.

### Slide 44 — Confusion structure

On where the errors fall: the confusions cluster between the two high-arousal classes, which points to arousal — not the sign of valence — carrying the separable axis. That's consistent with the pairwise analysis.

### Slide 45 — Per-trial permutation null

The strictest spatial control I ran: an independent channel permutation per trial, not one global shuffle. The classification still clears it. A flat classifier can't compensate for that.

### Slide 46 — Four documented pivots

And since methodology came up — these are the points where a result went against my hypothesis and I let it redirect the design rather than forcing the model through. The graph result is the clearest: I expected message passing to help, the theory said otherwise here, the data agreed, and that's what produced the design rule.

### Slide 47 — Coupling detail

The full coupling matrix behind the kappa number, if you'd like to see how the dynamical and topological families align electrode by electrode.

### Slide 48 — HC vs. MDD contrast

And the descriptor-level contrast between healthy controls and the depression group, for the clinical questions.

### Slide 49 — ERP-window dependence

If you want the window analysis: this is how discriminability depends on which part of the epoch we read, and it lines up with the early components rather than the late drift the black box leaned on.

### Slide 50 — Layers are complementary, not redundant

On whether the layers are just redundant: combining the temporal and spatial families never beats the better one alone, yet different disorders load on different families. That's the definition of complementary information, and it's why the decomposition is worth keeping.

### Slide 51 — Coupling by affective category

The coupling broken out by affective subcategory, if you want to see how the structure-function relationship shifts with the stimulus.

### Slide 52 — Layer-specific clinical sensitivity

And the full heatmap behind the disorder-layer table — which descriptor moves for which diagnosis, with substance use standing out, as the earlier slide showed.

### Slide 53 — Three generations of neural computation

If the framing comes up: first-generation units are threshold logic with no notion of time; second-generation deep nets abstract time away into firing rates; and the third generation — spiking — puts time back as the carrier. That's the family ARSPI-Net belongs to, and why it fits a signal whose information is in its timing.

