# M3-CoW: Mamba-3 rotational state, supervised as a state space, read as keys

Date: 2026-08-28 · Dataset104 (TopCoW joint CTA+MRA, 13 CoW classes), fold 0

---

## 1. Where we actually stand

Class-averaged over the 13 vessels, 50 held-out subjects:

| run | Dice | clDice | note |
|---|---|---|---|
| plain nnU-Net (joint) | **0.7044** | 0.8562 | the control |
| MF-SSM (A,C shared) | 0.6880 | 0.8465 | **below** the control |
| MF-SSM SharedAll (no factorisation) | 0.6846 | 0.8597 | factorisation worth +0.003 |
| MF-SSM CT-only → MRA zero-shot | 0.3957 | 0.4967 | |
| MF-SSM 5-shot adapter → MRA | 0.3957 | 0.4965 | **gained 0.0000** |
| MF-SSM MR-only → CTA zero-shot | 0.0022 | 0.0028 | total collapse |
| MR-only in-domain (MRA) | 0.7692 | 0.8829 | best single number we have |

TopCoW 2024 MRA Task 1, 1st place: **0.876** class-avg (single), ~0.901 ensemble.
Human ceiling ~0.902.

Two things follow. The seed-noise floor on this setup is 0.013, so the
factorisation's +0.003 is not evidence of anything. And an adapter that moves
Dice by 0.0000 across 4 decimal places is not underfitting — it has nothing to
adapt into.

## 2. Why, mechanically

**The state was never supervised.** With Dice/CE at the output, any `h` the
decoder can read is optimal. Nothing made `h` modality-agnostic, linearly
decodable, or non-degenerate. `B^MR` was fitted to write into a state space
whose geometry no loss ever constrained, so refitting it on 5 volumes changes
nothing that matters.

**`share_A=True` was nominal.** The update only ever sees the product
`dt^m · A`. `A` was tied while `dt_proj` stayed per-modality, so the *effective*
dynamics `exp(dt^CT A)` and `exp(dt^MR A)` were independent functions. "One
anatomy propagating" was not enforced by the parameterisation at any point.
This cannot be fixed structurally without giving up the modality-specific
timescale, which is the physically justified part — so it has to become a loss.

**A real diagonal state cannot hold the anatomy.** `a_t = exp(dt·A)`, `A<0`, so
`a_t ∈ (0,1)`: the state is a leaky accumulator that only forgets. Its class
prototypes end up ordered by how recently the scan visited them — a line. The
Circle of Willis is a **ring**, and 6 of 13 classes are **mirror pairs**. A line
cannot embed a cycle without tearing it, and the tear lands exactly where the
model fails today:

| on the ring, far apart in raster order | Dice | | off-ring / large | Dice |
|---|---|---|---|---|
| 3rd-A2 | 0.314 | | R-ICA | 0.851 |
| L-Pcom | 0.464 | | BA | 0.836 |
| R-Pcom | 0.472 | | L-ICA | 0.822 |
| Acom | 0.594 | | R-PCA | 0.773 |

## 3. What M3-CoW changes

### 3.1 Rotational state (`training/mamba3_ssm.py`)

`a_t = exp(-dt·λ)·(cos(dt·θ) + i·sin(dt·θ))` — decay **and** phase. Rotation is
periodic: the state returns to phase after `2π/(dt·θ)` tokens. That is the
mechanism that lets Mamba-3 solve state-tracking a real-diagonal SSM provably
cannot, and here it is what makes ring closure and bilateral symmetry
representable at all. S4D-Lin init spreads `θ_n = πn` so a bank of periods is
available from the start; `dt·θ` is tanh-limited to ±π (Nyquist — an aliased
phase is indistinguishable from a different frequency).

Trapezoidal input rule: `b_t = ½ dt B_t x_t + ½ a_t dt B_{t-1} x_{t-1}`, second
order rather than Euler's first. Still an affine recurrence, so the same
log-depth associative scan applies (10 steps for 560 tokens, not 560).

The scan is ours, not `mamba_ssm`'s, for a concrete reason: it **returns the
state trajectory and the realised dynamics**. The fused kernel returns neither,
and every loss below needs them. (`mamba_ssm` 2.3.1 is installed in `aneur` but
not in `nnunet_venv` — every MF-SSM run to date used the PyTorch fallback.)

### 3.2 Losses on the state space (`training/state_losses.py`)

| loss | attaches to | what it forces |
|---|---|---|
| `L_align` | `h` | a CT ICA state sits near an MR ICA state, far from a CT MCA state — via per-modality prototype banks, so **no paired scans needed** |
| `L_dyn` | `dt·λ`, `dt·θ` | the two modalities realise the **same spectrum** — turns nominal `share_A` into a real constraint |
| `L_probe` | `h`, `C` | a single linear layer recovers vessel class from the state alone |
| `L_vc` | `h` | variance + decorrelation; without it `L_align` collapses to a point |
| `L_ring` | prototypes | Gram matrix of the 10 ring classes matches `cos(φ_u − φ_v)` — the geometry is **literally circular** |
| `L_mirror` | prototypes | `P[R-x] − P[L-x]` is one shared vector across all 5 mirror pairs |

`L_align` is the training-time form of `evaluation/state_overlap.py`'s
`R = d(same class, cross modality) / d(diff class, within modality)`: the
statistic that was only ever *measured* is now the thing being *optimised*.
`L_ring` and `L_mirror` are the two a real state cannot satisfy — they are what
makes the complex `A` load-bearing rather than decorative.

### 3.3 State as keys (`training/anatomy_transformer.py`)

13 anatomy-anchored vessel queries cross-attend with **K = W_K h**. `L_probe`
and this are two halves of one design: attention retrieves by dot product, so
class information only recoverable through the decoder's nonlinearities is
information the attention cannot use. `L_attn` then trains query *c* to place
its attention mass on tokens labelled *c* — so "the state makes good keys" is a
trained property with a number on it, not a hope.

Queries also emit **per-vessel presence**, which the voxel decoder has no
mechanism for. That matters here specifically: R-Pcom is absent in 64/125
subjects, L-Pcom in 69/125, 3rd-A2 in 109/125 — the three worst classes.

Fusion is zero-initialised (as is the SSM residual `gamma`), so an untrained
M3-CoW is **bit-for-bit a plain nnU-Net** — verified, `max|Δ| = 0.000e+00`. Every
point of difference has to be earned.

## 4. Experiment matrix

Primary claim needs the ablation, not the headline number.

| trainer | tests |
|---|---|
| `nnUNetTrainerM3CoW` | full |
| `_NoStateLoss` | **the central control** — same architecture, Dice only |
| `_NoComplex` | real diagonal A (phase frozen) |
| `_NoComplexNoGeom` | real A, no ring/mirror — closest to Mamba-2 |
| `_NoTrapezoid` | Euler input rule |
| `_NoAlign` / `_NoDyn` | individual state losses |
| `_NoTransformer` | state not used as keys |
| `_CTonly` / `_MRonly` | single-modality source models |
| `_Adapt_n{1,2,5,10}` | adapter-only (37K params) vs `_full` (34M), same checkpoint, same n scans |

Predictions worth writing down before running: `_NoStateLoss` ≈ MF-SSM ≈ 0.688;
`_NoComplex` loses most on Acom/Pcom/3rd-A2 and on Betti-0; `_Adapt_n5` beats
`_Adapt_n5_full` at small n and both beat 0.3957.

### Diagnostics (`evaluation/state_geometry_m3.py`)

Run on every arm, because the ablation table alone cannot distinguish "the loss
helped" from "the loss helped for the stated reason":

| measurement | what it settles |
|---|---|
| linear-probe MODALITY accuracy | is the shared state actually shared, or two disjoint regions |
| ratio `R` | anatomy- or modality-dominated geometry (MF-SSM: 0.56, unoptimised) |
| linear-probe CLASS accuracy | state decodability — the precondition for keys |
| RING gram error | is the geometry circular; `_NoComplex` should not be able to lower it |
| MIRROR offset variance | is laterality one consistent displacement |

### One more cheap experiment worth running

**Modality-blind inference.** M3-CoW, like MF-SSM, needs `MFSSM_MODALITY` at
test time to select `B^m`. Once `L_align` and `L_dyn` have made the state
modality-independent, feeding the *wrong* modality id should degrade gracefully
rather than collapse — whereas in MF-SSM there is no reason it would. Running
each held-out CTA through `B^MR` and vice versa costs one inference pass and
directly evidences the central claim. If the drop is small, the modality label
stops being a requirement and becomes an optional hint, which removes the main
deployment objection to the whole line of work.

## 5. Getting to 0.876 — the honest part

The novelty above buys the cross-modality and topology claims. It will **not**
close a 0.17 Dice gap on its own, and no reviewer will believe it did. The gap
to 1st place is mostly engineering that the top TopCoW entries all did and we
have not:

1. **Two-stage ROI cascade** — biggest single lever. The CoW is ~2% of the
   volume; localise it, then segment inside the ROI at 0.3 mm. Currently we
   segment 0.5 mm whole-head patches where foreground is 0.10% of voxels.
2. **ResEnc-L plans** — nnU-Net itself now recommends these over the defaults
   we are on (the log says so on every run).
3. **5-fold + mirror TTA ensemble** — we have run fold 0 only, no TTA.
4. `mamba_ssm` in `nnunet_venv`, or accept the fallback scan's cost.

Realistic decomposition: cascade +0.06–0.09, ResEnc-L +0.01–0.02, 5-fold+TTA
+0.02–0.03, M3-CoW +0.02–0.04 concentrated in the rare ring classes. That lands
at or just past 0.876 — but the cascade is doing most of the arithmetic and the
paper's claim should be about the state space, evidenced by the ablation table
and the transfer study, not by the leaderboard row.

### 5.1 The cascade — built and preprocessed

`cascade/make_roi_dataset.py` + `cascade/predict_cascade.py`. Dataset107_CoW_ROI
is built, planned and preprocessed (250 cases, folds copied from Dataset104 so
the held-out subjects stay byte-identical).

**Stage 1 needs no training.** Measured on the 50 held-out cases: a 15 mm margin
around the predicted foreground box of an *already-trained* model (MF-SSM,
class-avg Dice 0.688) contains the full GT CoW in **50/50 cases**; 8 mm covers
96 %, 5 mm covers 80 %. Localisation is far easier than segmentation, so an
existing checkpoint is the localiser and a dedicated ~40 h run is skipped.

What the crop actually buys, measured rather than assumed:

| | Dataset104 | Dataset107 (ROI) |
|---|---|---|
| foreground fraction | 0.076 % | **0.493 %** (6.5× denser) |
| whole CoW inside one training patch | 70 % | **84 %** |
| in-plane spacing | 0.5 mm | **0.3525 mm** (1.42× finer) |
| a 1.5 mm Pcom, in-plane | 3.0 voxels | **4.3 voxels** |
| ROI as fraction of original volume | — | 14.9 % |

The resolution gain is real, not interpolation: native MRA is 0.297 mm in-plane
and native CTA 0.43 mm, so the current 0.5 mm isotropic preprocessing was
*discarding* resolution. nnU-Net's planner chose 0.60 × 0.3525 × 0.3525 mm from
the cropped data on its own. Patch (80, 160, 192) = 48 × 56 × 68 mm, against a
p90 CoW extent of 42 × 55 × 68.5 mm — so the ring is now essentially
unfragmented, which is what `L_ring` and `L_mirror` need to be scored on
anything but a fragment.

One label trap worth recording: the TopCoW release numbers the third A2 as
**15**, not 13 (verified across all 250 cases: release uses {0..12, 15}).
Dataset104's builder remaps it; `make_roi_dataset.py` reproduces that remap, or
the ROI dataset's classes would silently disagree with every existing result.

Verified: M3-CoW builds and trains on the Dataset107 plans — 600 key tokens at
the (5,10,12) grid, 0.68 s/iter, 7.57 GiB, untrained output identical to the
backbone to 0.000e+00.

Run stage 2 with:  `TRAINER=nnUNetTrainerM3CoW DATASET=107 GPU=n ./run_m3cow.sh`

**Still not done:** ResEnc-L plans, 5-fold, TTA, and the end-to-end cascade
evaluation. No number should be quoted as "beats TopCoW SOTA" until stage 2 has
trained and `predict_cascade.py` has been run end to end against the held-out
set.

## 5.2 RESULTS — first completed run (fold 0, 1000 epochs, 2026-08-30)

1000/1000 epochs, ~27 h, ~95 s/epoch, **zero NaN**; the non-finite guards never
fired once (`nonfinite 0.00, skipped 0.00`), so the fp32-scan / dt-clamp /
skip-instead-of-clip fixes addressed the cause rather than masking it.

Held-out set, 50 subjects, class-averaged over the 13 vessels:

| run | Dice | clDice | HD95 | Betti0 |
|---|---|---|---|---|
| **M3-CoW** | **0.7272** | **0.8732** | **3.53** | **0.196** |
| baseline nnU-Net | 0.7044 | 0.8562 | 3.98 | 0.286 |
| MF-SSM | 0.6880 | 0.8465 | 4.51 | 0.275 |
| MF-SSM SharedAll | 0.6846 | 0.8597 | 4.24 | 0.265 |
| CMAP | 0.6925 | 0.8503 | 3.67 | 0.242 |

+0.0228 over the baseline, above the 0.013 seed-noise floor. **The first method
in this project to beat plain nnU-Net** — every prior attempt sat below 0.7044.
Betti0 fell 0.286 -> 0.196 (31 % fewer spurious components).

Gains concentrate in the thin communicating segments, as the ring argument
predicts: R-Pcom **+0.140**, 3rd-A2 **+0.066**, L-ACA +0.029, R-ACA +0.026,
L-MCA +0.024. Large vessels are flat (BA -0.002, R-ICA -0.002). Two classes went
the wrong way: Acom **-0.017** and R-MCA -0.012. Acom is a ring class the
argument specifically named, so that is a genuine miss, not noise to wave away.

### Modality-blind inference

Forcing all 50 cases (including the 25 MRA) through `B^CT`:

| | Dice | clDice |
|---|---|---|
| correct modality per case | 0.7272 | 0.8732 |
| all forced through `B^CT` | 0.7270 | 0.8732 |
| degradation | **-0.0002** | +0.0001 |

Two trivial explanations were checked and ruled out:

* **The operators did not collapse onto each other.** `blk0.x_proj_B`:
  cos 0.78, rel.diff 0.66 (related but distinct); `blk0.dt_proj`: cos -0.009,
  i.e. as different as two independent random inits. They stayed genuinely
  different operators.
* **The SSM path is not inert.** Its residual `gamma` was zero-initialised and
  learned to **-0.509** in block 0.

So two materially different measurement operators write into the same state
region, which is precisely what `L_align` and `L_dyn` were built to enforce and
what MF-SSM could not do (there, MR-model -> CTA collapsed to 0.002 Dice).
Practically it means the modality label becomes an optional hint rather than a
deployment requirement.

### What the network chose to use (all these paths were zero-initialised)

| path | learned value |
|---|---|
| SSM block 0 gamma | **-0.509** |
| SSM block 1 gamma | -0.002 (**inert**) |
| transformer `w_prior` | norm 1.364, max 0.764 |
| transformer `w_presence` | norm 0.481 |
| `adj_scale` (anatomical bias) | -0.260 |

`w_presence` loads highest on 3rd-A2 (0.262), L-Pcom (0.186), R-Pcom (0.145),
Acom (0.149) -- exactly the often-absent classes the presence head was designed
for, which is a specific confirmation of that design rather than a global one.

**Actionable:** block 1 (the deepest, 140 tokens) is dead weight -- gamma
-0.002 and its `B` norm is 0.120 against 8.234 in block 0. `N_STAGES_TO_WRAP=1`
should cost nothing and save compute.

### What this does NOT establish

`_NoStateLoss` and `_NoComplex` have not run. The +0.023 cannot yet be
attributed to the state losses rather than to the architecture or the extra
3.6M parameters. One fold, one seed. And 0.7272 is still 0.149 short of the
0.876 TopCoW single-model figure.

## 6. Verified so far

- 39/39 unit tests (`training/test_m3cow.py`): scan matches a sequential complex
  recurrence to 4.8e-7; rotation is exactly periodic while a real state saturates
  and never returns; `L_dyn` is 0 iff spectra agree and backprops into `A_log`,
  `A_theta`, `dt_proj`; `L_ring` is 3.4e-15 on a perfect circle; `L_attn` is 0 at
  its optimum regardless of class size; AMP and checkpointing paths finite.
- Full 2-epoch run on real Dataset104 fold 0 (train 200 / val 50, split verified
  patient-disjoint): completed training + validation, loss 3.59 -> 2.88,
  294 s then 282 s per epoch on the pre-bottleneck config.
- `L_attn` is a KL, not a raw cross-entropy: the target's entropy log(count) is
  an irreducible offset that would have started the term at log(560)=6.3 and let
  it dominate early training at weight 0.3. Now zero at the optimum.
- End-to-end at real patch size (112×160×128, batch 2), 35.50M params
  (30.80 backbone + 1.44 SSM + 3.26 transformer), untrained output identical to
  the backbone to `0.000e+00`.
- Cost, against a 0.579 s/iter plain-nnU-Net reference on the same card:

  | config | s/iter | peak |
  |---|---|---|
  | full-width scan, grad-checkpointing on | 1.33 | 5.6 GiB |
  | full-width scan, checkpointing off | 1.07 | 8.9 GiB |
  | **bottleneck 128, checkpointing off** | **~0.79** | **6.6 GiB** |

  Checkpointing was buying 3 GiB we do not need on a 40 GiB card while paying a
  full recompute; the channel bottleneck is the better lever. Settled at ~1.4x
  the backbone. (Both cards were shared with other jobs during profiling, so
  these are indicative, not precise.)

### Epoch budget

The MF-SSM run's per-class pseudo-dice was still climbing at epoch 950 (best
0.7932 at 951; within 1% of best only from epoch 845), so nnU-Net's 1000-epoch
default is **not** past convergence here — that is the poly LR schedule doing
its work, not overfitting. Truncating is therefore not free. Protocol: run the
ablation matrix at **500 epochs** (a compressed, not truncated, LR schedule —
every arm pays the same cost so the comparisons stay valid) and the final
headline models at 1000.
