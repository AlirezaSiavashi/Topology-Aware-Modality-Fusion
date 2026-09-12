# Proposal Extension — Revision Worksheet
Source documents: `research_proposal_for_vessel_segmentation_2_CH.pdf` (original, with CH comments)
and `report_for_hpi.pdf` (2-month results report, 12.09.2026).

Strategy: keep the original proposal structure and the whole of Stage 2 unchanged.
Rewrite only Stage 1, insert one new "Progress to Date" section, re-anchor the H100
justification on measured evidence, and replace the project-plan table.

---

## PART A — Status audit of every proposal element

| # | Proposal element | Status | Evidence from the report |
|---|---|---|---|
| 1 | Mamba SSM backbone for Stage 1 | **DONE, redesigned** | Modality-factorised Mamba-3: shared dynamics A and readout C, per-modality measurement operator B and step size Dt |
| 2 | Direct supervision of the recurrent state | **DONE — new, not in the original proposal** | Six objectives; ring topology and bilateral symmetry imposed on the phase of the complex state |
| 3 | clDice topology-aware loss | **DONE** | clDice 0.886 |
| 4 | 3D Dynamic Snake Convolution | **EVALUATED, NOT ADOPTED** | ~45 GB VRAM runs; the factorised SSM outperformed it |
| 5 | Swin / window-attention comparison | **DONE** (not in original plan) | ~45 GB runs, outperformed by the factorised SSM |
| 6 | FiLM cross-attention decoder | **SUPERSEDED** | Replaced by an anatomy-anchored query decoder whose keys come from the SSM state |
| 7 | Voxel-spacing selection study | **DONE** (not in original plan) | 0.5 mm isotropic vs native 0.35 x 0.35 x 0.60 mm, full 1000-epoch runs each |
| 8 | Baselines under identical configuration | **DONE** | nnU-Net, CMAP, MF-SSM, MF-SSM SharedAll; identical config, split and schedule |
| 9 | Topology evaluation pipeline | **DONE** | Dice / clDice / HD95 / Betti-0, per-case class average over present classes |
| 10 | Paired statistical testing vs nnU-Net | **DONE for fold 0** | Seed-to-seed floor measured at 0.013 Dice |
| 11 | P1 deliverable "clDice > 0.85" | **MET (0.886)** — but at patch level, not with the stride = 1 navigator | Table 1 of the report |
| 12 | Full-volume navigator at stride = 1 | **NOT STARTED** — blocked by A100 40 GB | Section "Stage 1 computational constraints" |
| 13 | 5-fold cross-validation + ensemble | **NOT STARTED** — fold 0 only | |
| 14 | Replicate seeds / error bars | **NOT STARTED** | |
| 15 | Native-resolution second stage for Acom/Pcom | **NOT STARTED** | Weakest classes remain at 0.55-0.64 Dice |
| 16 | Ablation isolating Dt | **NOT RUN** | Objective 3 of the report explicitly not tested |
| 17 | Subject-disjoint (unpaired) validation | **IN PROGRESS** | Splits constructed, runs ongoing |
| 18 | RSNA 2025 CTA as the CTA source | **CHANGED** | TopCoW CTA used instead; RSNA now proposed for external validation only |
| 19 | Stage 2: variant identification from the mask | **NOT STARTED** | |
| 20 | Stage 2: 3D conditional Fast-DDPM | **NOT STARTED** | |
| 21 | Stage 2: Monte-Carlo uncertainty maps | **NOT STARTED** | |

**Headline for the extension: Stage 1 is de-risked but not finished. The segmentation
quality target was met; the resolution and the statistical-rigour targets were not.
Stage 2 is entirely ahead.**

---

## PART B — Text blocks to paste

### B1. Insert a new section "Progress to Date" immediately after *Approach* (~230 words)

> **Progress to Date**
>
> The first two months of the project delivered the Stage 1 segmentation model in a form
> that differs from, and improves on, the design originally proposed. Rather than relying
> on architecture alone, the recurrent state of the state-space model is now supervised
> directly. A single Circle-of-Willis state, defined by shared dynamics and a shared
> readout, is written to by modality-specific measurement operators, and six auxiliary
> objectives constrain that state, two of which impose the ring topology and the bilateral
> symmetry of the CoW on the phase of the complex-valued state.
>
> On 50 held-out, patient-disjoint volumes (25 CTA, 25 MRA) of the TopCoW dataset, the
> model attains a per-case class-average Dice of 0.786 against 0.765 for an identically
> configured nnU-Net, clDice of 0.886, HD95 of 3.09 mm and a Betti-0 error of 0.138. The
> Betti-0 error, which counts errors in the number of connected components, falls by 53 %
> relative to nnU-Net and by 41-45 % relative to other state-space variants. Removing only
> the auxiliary objectives, at identical parameter count, reduces Dice to 0.750, below the
> baseline, so the gain is attributable to what the state is trained to represent rather
> than to the added capacity.
>
> The clDice target of the original P1 milestone is therefore already exceeded. Two
> elements of Stage 1 remain: the sub-millimetre resolution needed for the communicating
> arteries, and the cross-validated statistical evidence expected by the target venues.
> Both are the subject of this extension.

---

### B2. Replace the paragraph beginning *"We train the segmentation stage on the combined TopCoW 2024 MRA and the RSNA 2025 Intracranial Aneurysm dataset..."*

Remove the old preliminary numbers (Dice 0.811 / clDice 0.821 / Betti-0 7.06 per case on
RSNA fold 0) — they are superseded. Replace with:

> We train the segmentation stage on the TopCoW 2024 CTA and MRA cohorts jointly, with the
> RSNA 2025 Intracranial Aneurysm dataset reserved for external validation. Our completed
> single-fold results (Dice 0.786, clDice 0.886, Betti-0 error 0.138 per case, without
> ensembling) already exceed the connectivity of every published state-space baseline we
> evaluated under identical conditions. The stride = 1 navigator, together with five-fold
> cross-validation and ensembling, is expected to raise mean Dice into the 0.82-0.85 range
> while preserving that connectivity advantage. This combination, competitive overlap
> accuracy together with connectivity preservation, is what enables the Stage-2 diffusion
> model to condition on a structurally faithful mask.

*(Note: I lowered the projected Dice band from the original 0.85-0.87 to 0.82-0.85. The
original figure was extrapolated from an RSNA binary-vessel run; the 13-class TopCoW task
measured here sits lower, and promising 0.85-0.87 on a 13-class metric would be very hard
to defend at review.)*

---

### B3. Rewrite the "three coupled innovations" list in Stage 1

Item (2) Dynamic Snake Convolution and item (3) FiLM must change — one was tested and
dropped, the other was replaced. Suggested replacement for the opening paragraph of
*Stage 1*:

> Connectivity is pursued through three coupled mechanisms. (1) Direct supervision of the
> recurrent state, so that the ring topology and the bilateral symmetry of the CoW are
> imposed on the representation itself rather than only on the segmentation output; this
> component is implemented and evaluated, and accounts for the reduction in connected-
> component error reported above. (2) An anatomy-anchored query decoder that reads the
> supervised state, one query per vessel class, replacing the feature-wise modulation
> originally envisaged. (3) A full-volume navigator that processes the entire brain in a
> single forward pass, so that no vessel crossing a patch boundary is seen as two
> independent fragments; this component remains to be built and is the principal subject of
> the requested extension. Tubular-geometry convolutions were evaluated as an alternative to
> (1) and did not match it at equal cost, and are therefore not carried forward.

---

### B4. Re-anchor the stride = 1 justification (item (i) of the H100 section)

The original argument was "patch-based processing breaks topology". Your own results now
partly answer that argument — a patch-based model reached Betti-0 0.138 — so a reviewer
could turn it against you. Re-anchor it on the measured per-class deficit instead, which
is stronger because it is evidence rather than assertion. Add after the existing stride
explanation:

> Our completed experiments localise the remaining error precisely. Overall connectivity is
> already good (Betti-0 error 0.138 per case), but the residual failures are concentrated in
> the communicating arteries, which remain at 0.55-0.64 Dice while the large vessels exceed
> 0.85. These are the vessels of about one millimetre in caliber, and they are also the ones
> that determine collateral reserve and therefore matter most to Stage 2. The deficit is
> therefore a resolution limitation rather than a context limitation, and it is exactly what
> stride = 1 tokenisation addresses. This is the single best-evidenced reason for the
> requested memory headroom.

### B5. Strengthen the 5-fold justification (item (ii)) with measured numbers

> Single-fold variance is high on our data: the per-patient standard deviation of the
> completed run is 0.088, and the measured seed-to-seed variation is 0.013 Dice, against an
> improvement over the baseline of 0.021. A single fold and a single seed are therefore not
> a sufficient basis for the claims the target venues expect, and this is currently the main
> limitation of the completed work.

### B6. One sentence to add to the throughput argument

> The value of the H100 for this project is already demonstrated rather than projected: the
> voxel-spacing study that fixed the preprocessing grid required several complete
> 1000-epoch runs compared on a like-for-like basis, and was made feasible within the
> project period by the mixed-precision throughput of the H100 rather than by its capacity.

---

### B7. Replace the Project Plan table

> The extension spans 12 months in three phases. Phase P0 is complete and is reported
> separately; Stage 1 is now partially de-risked by measured results, and Stage 2 remains
> the novel high-risk contribution and the primary consumer of GPU memory.

| Phase | Months | Objective | Deliverable | H100 footprint |
|---|---|---|---|---|
| **P0 — completed** | (done) | Modality-factorised state-space segmentation with direct state supervision; voxel-spacing study; baselines under identical configuration | Dice 0.786, clDice 0.886, Betti-0 error 0.138 on 50 held-out volumes; clDice milestone exceeded | delivered on the existing allocation |
| P1 — resolution | 3 | Full-volume navigator at single-voxel token resolution; native-spacing second stage targeting the communicating arteries | Dice above 0.70 on the anterior and posterior communicating arteries | 45 GB (infeasible on A100 40 GB) |
| P2 — validation | 2 | Five-fold cross-validation, three replicate seeds, subject-disjoint splits, ablation isolating the learned step size | Cross-validated result with error bars; five-checkpoint ensemble; per-class Dice | 50 GB, batch size 2, approx. 12-15 GPU-days |
| P3 — post-occlusion synthesis | 7 | 3D conditional Fast-DDPM trained on the congenital-variant cohort; Monte-Carlo uncertainty maps | Counterfactual post-operative angiography with voxel-wise variance maps; FID <= 30, SSIM >= 0.82, ECE <= 0.05 | 70-75 GB peak |

---

## PART C — Reviewer comments from the previous round (fix before resubmitting)

| Ref | Comment | Action |
|---|---|---|
| CH1 | "Needs a citation" (balloon test occlusion) | Add a clinical BTO reference in the abstract |
| CH2, CH3 | Footnote links | Already rendered; verify they survive the export |
| **CH4** | **"How can you guarantee this? Explain in the text."** | **Most important. You cannot guarantee it, and your own Betti-0 of 0.138 is not zero. Remove the word "guaranteed" everywhere — title of Stage 1, abstract, figure. Replace with "connectivity-preserving" or "topologically faithful", and state the measured Betti-0 as the evidence. This turns the weakest point of the previous version into its strongest.** |
| CH5 | Word order, "proposed" | "Our proposed segmentation backbone..." |
| CH6 | Expand abbreviation (SSM) | Expand on first use |
| CH7 | "To" -> "to" | Typo |
| CH8 | "Step" | Typo |
| CH9 | "What is the stride needed for?" | Answered in the existing text; B4 strengthens it further |
| CH11 | Expand (DDPM) | Expand on first use |
| CH12 | "by H100 VRAM size of 80 GB" | Rephrase as suggested |
| CH13 | Undefined terms in the table (NavBrush, Betti, MedNeXt) | Betti-0 is now defined in the new Progress section, so it may stay. Decide whether to keep the name NavBrush at all: the architecture has changed enough that dropping the name and describing the components is cleaner. Remove MedNeXt |
| CH14 | Undefined term | Same treatment |

---

## PART D — Three honest risks a reviewer may raise

1. **The architecture changed.** The proposal promised a full-volume navigator with snake
   convolutions and FiLM; what was built is a patch-level factorised SSM with state
   supervision. Present this as a finding, not a deviation: the state objectives delivered
   the connectivity the navigator was meant to deliver, at lower cost, and the navigator is
   now needed for resolution rather than for context.

2. **The dataset changed.** RSNA CTA was replaced by TopCoW CTA. Justify it: TopCoW carries
   13-class labels in both modalities, which is what the modality-factorisation claim
   requires; RSNA does not, and is better used as external validation.

3. **One fold, one seed.** State this plainly rather than letting a reviewer find it, and
   make it the explicit purpose of phase P2.
