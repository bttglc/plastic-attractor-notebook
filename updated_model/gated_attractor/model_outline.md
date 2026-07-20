# gated_attractor model outline

Reference for this package. Gate non-selectivity is fixed (section 9);
section 13 traced the accuracy ceiling to unstable stimulus-to-conjunction-
unit routing, and section 14 confirms it — making the slow plastic-weight
component dominate W nearly closes the gap (real-block accuracy 0.565 ->
0.881, `2cpr_slowW3`, current best version). Section 15 finds this fix is
gating-specific, not a general routing-instability cure: it does nothing for
`cued_attractor` (no gate) and actively hurts `gated_simple_attractor`
(oracle suppression, already near ceiling under the old fast-dominant W).

## 1. Lineage

Extends Whyte et al. 2025, 'A plastic attractor model of flexible rule-based
selective attention' (bioRxiv 2025.09.07.674747; Zotero 7L2R5LYQ), built on
Manohar et al. 2019's plastic attractor architecture. `model.py`'s core
equations are near-verbatim from `published_model/plastic_attractor/model.py`;
this package adds cued task-switching on top: the network is taught once (two
instruction blocks), then runs real blocks where the rule varies trial-by-trial
and is signalled only by a cue.

## 2. Neural populations

Defined in `model.py`'s `PlasticAttractor`:

- **Feature units** (10 default, `num_cues_per_rule=2`): green, blue, square,
  circle, 2 colour-cue, 2 shape-cue, action1, action2. Colour/shape/cue/action
  each form a competing group with within-group lateral inhibition
  (`feature_lateral_weight=-0.28`) and self-excitation
  (`feature_self_weight=0.73`).
- **Conjunction units** (4 default): global winner-take-all
  (`conjunction_lateral_weight=-0.45`, `conjunction_self_weight=1.00`),
  coupled to feature units via the plastic weight matrix W.
- **Gating units** (0 or 2, one per rule): optional feedforward inhibitory
  interneurons, driven by the cue, meant to suppress the task-irrelevant
  dimension. 0 disables them, reproducing Whyte et al. bit-for-bit. Also
  winner-take-all (`gating_lateral_weight=-0.45`, `gating_self_weight=1.6` —
  see 'gate persistence' below).

## 3. Task / vocabulary (`task.py`)

- `Feature`: GREEN, BLUE, SQUARE, CIRCLE (indices 0-3).
- `Task`: COLOR, SHAPE — the rule. `relevant_feature`/`irrelevant_feature`
  pick the task-relevant/irrelevant dimension of a `Stimulus`.
- `Vocabulary` lays out cue/action indices after the 4 fixed features.
  `gate_target_indices` hardcodes, per gate, which feature rows it may ever
  suppress — its task's irrelevant pair only. This is the fix for gate
  non-selectivity (section 9).
- Stimuli are congruent when colour and shape indicate the same response
  (green+square, blue+circle).

## 4. Dynamics (`model.py::PlasticAttractor.step`)

```
gating_inhibition = gating_to_feature_gain * (gating_output_weights @ (gate_activity - baseline))   # 0 if gating off
next_features      = clip(baseline + feature_recurrent @ centered_features
                           + conjunction_to_feature_gain * (W @ centered_conjunctions)
                           + external_input - gating_inhibition, 0, 1)
next_conjunctions  = clip(baseline + conjunction_recurrent @ centered_conjunctions
                           + feature_to_conjunction_gain * (W.T @ centered_features)
                           + noise, 0, 1)
next_gates         = clip(baseline + gating_recurrent @ centered_gates
                           + cue_to_gating_gain * (gating_input_weights @ centered_cues)   # only while cue_signal_active
                           + gate_external_input, 0, 1)                                    # forced drive during instruction
```

`centered_x = x - baseline_activity` (0.175) throughout — every recurrent/
plastic term is covariance-style. All three populations update synchronously.
`clip(., 0, 1)` is why suppression can't act while a feature is externally
floored at 1.0 (section 10).

**Gate persistence.** `gating_self_weight=1.6 > 1` gives the 2-gate
winner-vs-loser mode an eigenvalue > 1: a brief cue-driven push during
`stimulus_window` latches one gate near ceiling for the rest of the trial.
Cue input is only read while `cue_signal_active` (scoped to
`stimulus_window`) — otherwise bottom-up conjunction feedback could re-light
the wrong cue mid-trial.

## 5. Learning rules

**Main W (feature <-> conjunction), `_update_plastic_weights`:** raw,
per-step, unsupervised Hebbian covariance, no reward gating, exactly as
published:
```
change = outer(centered_features, centered_conjunctions)
change[cue_indices] = 0        # cues only reach the network via gating
fast_weights = clip(fast_weights + fast_learning_rate * change, 0, 1.0)     # gamma1=0.02
slow_weights = clip(slow_weights + slow_learning_rate * change, 0, 0.2)     # gamma2=0.0002
W = fast_weights + slow_weights
```
Fast/slow split is Whyte et al.'s addition: without the slow, narrowly-bounded
component, conjunction-unit selectivity drifts block to block and the
congruency effect disappears.

**Gating weights (cue -> gate input, gate -> feature output):**
three-factor / eligibility-trace rule (Fremaux & Gerstner 2016):
1. `_accumulate_gating_trace` builds a decayed trace (`gating_trace_decay=
   0.98`) only while `cue_signal_active`.
2. The trial runs under fixed gate weights; nothing applied until the outcome
   is known.
3. `consolidate_gating_trace(reward)`, once per trial, applies the trace
   scaled `+1`/`-1` by whether *the gate itself* won the correct side
   (`gate_winner_matches_task`) — not overall trial correctness, which would
   let unrelated W/congruency errors erode an already-correct gate mapping.

Output weights are also masked (`_gating_output_mask`, from
`gate_target_indices`) at init and every trace step, so a gate's own
dimension can never accumulate nonzero trace or weight.

## 6. Trial timing (`experiment.py::EpochProtocol`)

400 steps/trial: `[0, 51)` ISI (input=-1); `stimulus_window=[51, 101)`
stimulus+cue at 1.0; `teaching_window=[50, 351)` (instruction only) forces
action units +-1; `response_window=[101, 351)` zero input, network settles,
`response_search_start=110` scans for a 98%-of-peak crossing on action units;
`[351, 400)` ISI.

Stimulus and cue are always presented together at full strength regardless of
gating — suppressing the irrelevant one is entirely the gate's job.

## 7. Experiment structure (`run_switching_experiment`)

1. **Instruction** (`num_practice_blocks=2`): every cue x stimulus
   combination, response forced via `teaching_window`, gate forced via
   `_gate_drive_schedule` (+1 correct gate/-1 other, during
   `stimulus_window`).
2. **Performance practice**: one block/rule, unforced, learning on.
3. **Real/switching blocks**: one per `switch_probs` entry, rule varies
   trial-to-trial, signalled only by the cue; learning stays on throughout
   (`learn_during_trials`).

`model_versions_config.py`'s `2cpr_gating_units` is the only preset with
gating on (`number_of_gating_units=2`, gains per section 11).

## 8. Diagnostics (`analysis.py`, `launcher.py`)

`launcher.py` runs a version across `n_seeds` (`ProcessPoolExecutor`),
reduces via `analysis.py`, writes `output/<version>/simulation_data.npz` +
figures. Collects accuracy/RT/congruency/switch-cost plus
`gate_accuracy_by_block`, `no_response_rate_by_block`,
`colour_shape_row_norms_by_block`, `relevant_irrelevant_activity_by_kind`,
and (section 13) `conjunction_routing_flip_rate_by_block` /
`conjunction_routing_drift_by_kind`, plotted in
`conjunction_routing_stability.png`. Same `_SETTLED_ACTIVITY_WINDOW_STEPS=20`
convention used in `gated_attractor`, `gated_simple_attractor`, and
`cued_attractor` alike.

## 9. Fixed: gate non-selectivity

**Symptom:** `test_practice_teaches_above_chance_behaviour` failed (0.458
accuracy) before this fix.

**Cause:** both dimensions were active with the correct gate throughout every
`stimulus_window` regardless of relevance, so the eligibility trace had no
signal to prefer one — gate output weights saturated symmetrically on both.

**Fix:** hardcoded structural masking (`gate_target_indices` +
`_gating_output_mask`, section 5), not learned. Verified: own-dimension
weight always 0.0; gate winner matches the true task 100% of checked runs;
the test now passes.

## 10. Open question: selectivity fixed, behaviour not clearly better

**Structural finding:** suppression is arithmetically incapable of acting
while the stimulus is on screen — the irrelevant feature is externally
floored at 1.0 during `stimulus_window`; max possible suppression at
`gating_to_feature_gain=0.4` (~0.396) is under half of what's needed to pull
the `clip(0, 1)` sum below 1.0. Confirmed empirically: irrelevant-feature
minimum during `stimulus_window` measured exactly 1.000 in every trial/seed
checked.

**Consequence:** `_update_plastic_weights` runs every step including
`stimulus_window`, so the fully-active irrelevant feature gets Hebbian-bound
to whatever conjunction unit is active — confirmed in W (some units end up
with near-equal weight on both members of a competing pair).

**Tried:** excluding the winning gate's target rows from W's update during
`stimulus_window` + a ~10-step buffer. Improved raw within-dimension weight
discrimination relative to blunter pause variants, but real-block accuracy
didn't move (~0.51 vs ~0.495, 12 seeds) and the practice test regressed.
Section 11 found this was the wrong baseline: vs. no pause at all, the
mechanism left selectivity worse, until combined with gain re-tuning.

**Untried candidate:** present the cue before stimulus onset (cue-stimulus
interval, cf. Rogers & Monsell 1995, Meiran 1996) so suppression has already
engaged before the stimulus floors the irrelevant feature — a bigger
structural change than a parameter tweak.

## 11. Plasticity pause merged, gains re-tuned, measurement corrected

**Plasticity pause merged** into the package (was
`wip_plasticity_pause_weight_mask/`, now just how `_update_plastic_weights`/
`run_epoch` work; only active when gating is on).

**Corrected measurement:** max-gap checks hid real per-unit failures. New
`analysis.py` functions (`conjunction_unit_discrimination`: per-unit gap, not
maxed; `accuracy_by_task`/`accuracy_by_task_by_kind`) applied to a
pause-vs-no-pause comparison (n=20, `2cpr_gating_units`) showed the pause's
earlier "improvement" only held relative to blunter pause variants — vs. no
pause at all, it left selectivity worse (well-separated units 1.20/4 ->
0.10/4), accuracy flat (~0.51). Now standard practice: check all conjunction
units, never a single max/pooled number.

**New mechanism:** fixed multiplicative relevant-feature enhancement
(`gating_to_relevant_feature_gain`). First (additive) version was actively
harmful — inflated both members of the relevant pair equally, collapsing
discrimination. Fixed to multiplicative
(`gain * gate_deviation * centered_features`), amplifying only the
already-driven member. Default `0.0`, exact no-op.

**Retuned** via grid + fine sweep (n=8 then n=10, uncommitted scratch
scripts): inhibition alone raised well-separated-unit count monotonically
without moving accuracy; enhancement on top of moderate inhibition is what
moved accuracy, non-monotonically (0.8 reliably worse than 0.7). Chosen,
validated at n=20: `gating_to_feature_gain=0.7`,
`gating_to_relevant_feature_gain=0.6`.

| | gating off | old gains (0.4/0.0) | tuned (0.7/0.6) |
|---|---|---|---|
| real-block accuracy | 0.506±0.018 | 0.510±0.006 | **0.565±0.028** |
| colour / shape accuracy | 0.516/0.498 | 0.524/0.496 | 0.585/0.541 |
| well-separated units (/4) | 1.20 | 0.10 | **2.00** |
| amplifying eigenvalues (last block) | 2.60 | 2.15 | 2.55 |

First configuration where accuracy, colour/shape balance, and structural
selectivity all improve together at a trusted seed count.
`model_versions_config.py`'s `2cpr_gating_units` and the practice test now
use these values (test scenario: 0.417 -> 0.792, seed 0). `ModelParameters`
class defaults stay 0.4/0.0 (backward-compatible).

**Unresolved:** why 0.8 specifically breaks accuracy while 0.7/0.9 both work
wasn't chased down. Colour accuracy edges out shape (0.585 vs 0.541) — watch
if it's a real asymmetry. Full launcher figures regenerated in
`output/2cpr_gating_units/` with tuned values (section 12).

## 12. Relevant-feature saturation is near-ceiling; the gap is downstream

`relevant_irrelevant_activity_by_kind.png` (20 seeds, tuned gains) shows
settled relevant activity ~0.99-1.00, irrelevant ~0.001-0.009, flat across
switch-probability conditions. Confirmed in `simulation_data.npz`: relevant
exactly 1.0 in 79/80 seed x kind cells, irrelevant <0.01 in all 80.

**Mechanism:** once a gate latches (deviation ~0.825), the relevant
feature's effective self-gain becomes `feature_self_weight (0.73) +
gating_to_relevant_feature_gain (0.6) x gate_deviation (~0.825) ~= 1.22` — a
supra-unity feedback loop that runs to the `clip(0, 1)` ceiling by
construction (same latching principle as `gating_self_weight` itself,
section 4). The irrelevant feature has no such boost and decays to floor
under sustained suppression.

This sharpens, not resolves, section 10: with the feature layer this clean
(colour/shape separation ~0.99 vs ~0.005) but accuracy only ~0.565, remaining
errors must originate in conjunction-unit competition or the W action-row
mapping, not feature discrimination.

## 13. Where the gap lives: unstable stimulus-to-conjunction-unit routing

Scratch diagnostic (12 seeds, tuned params, 4608 real trials, uncommitted)
recorded per trial: winning conjunction unit, its W action-row differential,
and whether it favours `trial.correct_response`.

**Winner-take-all is decisive:** winning unit's share of settled activity
averages 0.992 (min 0.502) — not a source of ambiguity.

**Trial correctness tracks almost perfectly** whether the winning unit's
action-row favours the right response: 98.1% correct when it does (n=2764),
2.9% when it doesn't (n=1844) — matching pooled accuracy (0.600) almost
exactly. Downstream measurement/congruency/switch-cost explain little.

**Wrong-favouring trials aren't near-ties:** `|W[action1,winner] -
W[action2,winner]|` has the same distribution (median 1.115) regardless of
correctness; 92.9% of wrong-favouring trials have gap >=0.3 — each unit's
mapping is confidently learned, it's just sometimes the wrong unit winning.

**Which unit wins is unstable:** grouped by (seed, stimulus), 0/96
combinations keep the same winner across every real block in a run. Within a
single block, 55.3% of (seed, block, stimulus) groups show both correct- and
wrong-favouring outcomes across repeated presentations — routing flips trial
to trial, not just block to block.

**Reading:** unstable stimulus-to-conjunction-unit binding under continuous
plasticity, not weak competition or an ambiguous mapping.
`learn_during_trials=True` keeps fast Hebbian learning (no decay) running
every step of every real trial, so nothing holds a stimulus's routing fixed
after instruction — drift can displace a correctly-routed stimulus onto a
unit already wired for a different, opposite-action stimulus, producing a
confidently wrong response. Sharpens section 10's "capacity is tight by
construction": 4 units for up to 4 S-R mappings leaves no spare unit to
absorb displacement.

**Not yet checked:** whether this instability also occurs with gating off
(base-architecture property vs. gating-specific); correlation with trial
position within a block; whether `slow_weights` (capped 0.2) is failing to
anchor or is simply outweighed by the faster component.

**Promoted to tracked diagnostics** across all three `updated_model`
packages: `conjunction_routing_flip_rate_by_block` and
`conjunction_routing_drift_by_kind`, now in `analysis.py`/`launcher.py`/npz
export and `conjunction_routing_stability.png`, using the same settled-window
convention in `gated_attractor`, `gated_simple_attractor`, and
`cued_attractor` — lets the "gating off" comparison run directly
(`cued_attractor` has no gating; `gated_simple_attractor` is the
oracle-suppression control). Not yet run at a trustworthy seed count.

## 14. Slow-weight-dominant W all but closes the gap: 0.565 -> 0.881

`model_versions_config.py`'s `2cpr_slowW`/`2cpr_slowW2`/`2cpr_slowW3`/
`2cpr_slowW4` test section 13's last "not yet checked" item directly: is
`slow_weights` failing to anchor routing, or just outweighed by the faster
component? `W = fast_weight_blend*fast_weights + slow_weight_blend*
slow_weights` (`model.py:780-781`); every field below is unchanged from
`2cpr_gating_units` except `maximum_fast_weight`/`maximum_slow_weight`
(inverted from 1.0/0.2 to 0.2/1.0 across the four variants) and, for
`2cpr_slowW4` only, `slow_weight_blend` (1.0 -> 1.5). All four already run
at n=20 seeds via `launcher.py` (not scratch); numbers below are read
straight from `output/<version>/simulation_data.npz`.

| | `gating_units` (1.0/0.2) | `slowW` (0.7/0.7) | `slowW2` (0.5/1.0) | `slowW3` (0.2/1.0) | `slowW4` (0.2/1.0, blend 1.5) |
|---|---|---|---|---|---|
| real-block accuracy | 0.565±0.028 | 0.843±0.010 | 0.842±0.025 | **0.881±0.019** | 0.879±0.024 |
| congruent accuracy | 0.568-0.618 | - | - | 0.979-0.991 | 0.979-0.991 |
| incongruent accuracy | 0.527-0.566 | - | - | 0.757-0.790 | 0.760-0.784 |
| routing flip rate, real blocks | (not tracked) | (not tracked) | (not tracked) | 0.013-0.053 | 0.000-0.032 |
| routing drift rate, by kind | (not tracked) | (not tracked) | (not tracked) | 0.050-0.087 | 0.025-0.044 |
| colour / shape W-row gap (final) | 0.639 / 0.635 | - | - | 0.737 / 0.808 | 1.035 / 1.125 |
| amplifying eigenvalues (real, last block) | ~2.5 | - | - | ~2.5 | ~3.0 |

Answer: outweighed, not failing. Capping fast at 0.2 and slow at 1.0 (vs the
reverse) drops real-block routing flip rate roughly 10-40x relative to
section 13's old-config scratch numbers (55.3% of (seed, block, stimulus)
groups flipping mid-block), and real-block accuracy rises to within reach of
the near-ceiling feature-layer saturation section 12 already established.

**Congruency is the cleanest confirmation.** Under `2cpr_gating_units`,
congruent (0.57-0.62) and incongruent (0.53-0.57) accuracy were both
mediocre and barely distinguishable — routing noise was hitting every trial
regardless of congruency, swamping any real selective-attention signal. Under
`2cpr_slowW3`, congruent accuracy is near-ceiling (0.98-0.99) and incongruent
shows a real, bounded cost (0.76-0.79) — once routing stopped flipping, the
irrelevant-feature interference the model is meant to capture became visible
instead of being buried in routing lottery noise.

**Ruled out:** gate accuracy is 1.000 and no-response rate 0.000 in
`gating_units`, `slowW3`, and `slowW4` alike — none of this tracks back to
the gate (section 9's fix) or to response failures. Relevant/irrelevant
settled-activity saturation (section 12) is unchanged, ~1.0/~0.0 throughout.

**slowW3 vs slowW4 — a plateau, not a monotonic trend.** slowW4 pushes
slow-dominance further via `slow_weight_blend=1.5` (cap unchanged from
slowW3). It gets a *lower* routing flip rate, *larger* W-row gaps, and *more*
amplifying eigenvalues than slowW3 — yet real-block accuracy is statistically
tied (0.879±0.024 vs 0.881±0.019). More structural separation and stability
past this point stops buying more accuracy.

**Per-seed spread (`slowW3`):** min 0.734, max 1.000 across 20 seeds — a
real spread but no seed near chance, unlike the wide failure mode the old
config's routing instability could plausibly produce.

**Not yet checked:** a full-session stability count (all 8 real blocks
pooled, not just the ~2 sharing a switch probability that
`conjunction_routing_drift_by_kind` groups by) directly comparable to
section 13's "0/96" headline number; a finer cap sweep between `slowW2` and
`slowW3` to locate the plateau; whether `gating_to_feature_gain`/
`gating_to_relevant_feature_gain` (still 0.7/0.6, tuned in section 11 under
the old fast-dominant W) are still accuracy-optimal under slow-dominant W;
the same cap inversion on `cued_attractor` (no gating) and
`gated_simple_attractor` (oracle suppression), to test whether slow-weight
dominance is a general fix for tight-capacity routing instability or
gating-specific; whether `slowW3`'s lower-accuracy seeds (0.73-0.87) are
exactly the seeds with elevated routing flip rate.

## 15. Whole-session stability, a finer cap sweep, and cross-package generalization

Runs section 14's three "not yet checked" items together: a new
`conjunction_routing_flip_rate_full_session` (`analysis.py`, all three
`updated_model` packages) pools every real trial across all 8 blocks rather
than the ~2 sharing a switch-probability kind; a finer `gated_attractor` cap
sweep locates the accuracy/stability plateau; and the same 1.0/0.2 ->
0.2/1.0 fast/slow cap inversion is applied to `cued_attractor` and
`gated_simple_attractor` to test whether slow-weight dominance is a general
routing fix or specific to gating. All runs n=20 seeds via `launcher.py`.
`cued_attractor`/`gated_simple_attractor` had `practice_permutation_repeats`
left at 1 (vs. `gated_attractor`'s 5) going into this session — an
inconsistency, not a deliberate choice, fixed to 5 for every run below and
in both launchers going forward.

**Whole-session stability, directly comparable to section 13's "0/96":**
under `2cpr_gating_units` (old fast-dominant W), all 20/20 seeds have
full-session flip rate exactly 1.000 — literally no (task, stimulus)
identity anywhere stays on the same conjunction unit for an entire session.
Under `2cpr_slowW3`, mean flip rate drops to 0.119±0.026, and 7/20 seeds
reach perfect whole-session stability (0.000). Sharper than, and consistent
with, section 13's original "0/96" scratch estimate.

**Finer cap sweep** (`2cpr_slowW2_cap01/03/04`: `max_fast_weight` =
0.1/0.3/0.4, `max_slow_weight=1` throughout, alongside the existing 0.2
(`slowW3`) and 0.5 (`slowW2`) points):

| max_fast_weight | 1.0 (`gating_units`) | 0.5 (`slowW2`) | 0.4 | 0.3 | 0.2 (`slowW3`) | 0.1 |
|---|---|---|---|---|---|---|
| real-block accuracy | 0.561±0.025 | 0.842±0.025 | 0.858±0.024 | 0.870±0.021 | **0.881±0.019** | 0.872±0.024 |
| congruent accuracy | 0.547-0.621 | - | 0.951-0.960 | 0.976-0.982 | 0.979-0.991 | 0.957-0.972 |
| incongruent accuracy | 0.517-0.607 | - | 0.736-0.781 | 0.749-0.775 | 0.757-0.790 | 0.770-0.797 |
| full-session routing flip | 1.000±0.000 | n/a* | 0.237±0.026 | 0.150±0.026 | 0.119±0.026 | 0.100±0.029 |

*`slowW2` predates `conjunction_routing_flip_rate_full_session` and wasn't
rerun this session, so only its (pre-existing) accuracy is available.

Accuracy plateaus across the whole 0.1-0.5 range — every point from 0.842 to
0.881 sits within about 1 SE of every other, so `slowW3`'s 0.2 isn't a sharp
optimum, just the best-performing point sampled on a flat top. Full-session
routing flip rate, however, keeps falling monotonically as the fast cap
drops further (23.7% -> 15.0% -> 11.9% -> 10.0%) — the same "structural
stability keeps improving after accuracy stops" pattern section 14 found
between `slowW3` and `slowW4` at the high-dominance end, now confirmed at
the low end too. Congruent/incongruent accuracy is essentially identical
across 0.1-0.4, so this is a genuine plateau, not fragile tuning around one
lucky value.

**Cross-package generalization — does the cap inversion transfer?** Same
1.0/0.2 -> 0.2/1.0 inversion applied to each sibling package's baseline
config (`whyte_params_2cpr` in `cued_attractor`; `baseline` in
`gated_simple_attractor`), n=20, `practice_permutation_repeats=5`:

| | `cued_attractor` baseline | `cued_attractor` cap-inverted | `gated_simple_attractor` baseline | `gated_simple_attractor` cap-inverted |
|---|---|---|---|---|
| real-block accuracy | 0.512±0.026 | 0.535±0.023 | **0.952±0.032** | 0.491±0.048 |
| congruent accuracy | 0.515-0.531 | 0.542-0.562 | 0.949-0.957 | 0.465-0.506 |
| incongruent accuracy | 0.481-0.517 | 0.496-0.528 | 0.946-0.954 | 0.478-0.501 |
| full-session routing flip | 0.463±0.096 | 0.688±0.045 | 0.450±0.049 | 0.163±0.034 |
| no-response rate | 0.000 | 0.000 | 0.000 | 0.000 |

Neither sibling package replicates `gated_attractor`'s result, and in
opposite ways. `cued_attractor` (no gate at all) stays at chance either way —
congruent and incongruent accuracy are indistinguishable in both configs, and
cap-inversion makes full-session routing *less* stable (46.3% -> 68.8%), not
more. `gated_simple_attractor` (oracle suppression: the irrelevant feature is
externally forced off, not learned) is the sharper result: its existing
fast-dominant baseline is already near ceiling (0.952) despite a routing
flip rate (0.450) that would have crushed `gated_attractor`'s old
fast-dominant config — apparently harmless here, since oracle suppression
removes the interference that made routing identity matter in the first
place. Cap-inversion *lowers* its flip rate further (to 0.163) but collapses
accuracy to chance (0.491), congruent and incongruent alike (no-response
rate rules out a response-generation failure).

**Reading:** section 14's fix rescues a specific failure mode — a gate that
suppresses the irrelevant dimension well enough to create a real routing
target, but not perfectly, so continuous fast Hebbian learning can knock a
stimulus off that target. Without a gate at all (`cued_attractor`) there is
no clean target to anchor onto in the first place. With oracle suppression
(`gated_simple_attractor`) there was never a target to lose — and slow-
dominant W's slower plasticity apparently tracks the real-block rule-switch
structure worse than the fast component did, once there's no gating noise
for it to be anchoring against. Not a general routing-instability cure —
gating-specific.

**Repeats-for-teaching sweep**, on `cued_attractor`'s cap-inverted config
(same config as above; `practice_permutation_repeats` = 1/3/5/10, n=20
each — 1 was this package's pre-existing, apparently-accidental default):

| repeats | 1 | 3 | 5 | 10 |
|---|---|---|---|---|
| real-block accuracy | 0.493±0.036 | 0.482±0.033 | 0.535±0.023 | 0.510±0.010 |
| congruent accuracy | 0.509-0.535 | 0.481-0.511 | 0.542-0.562 | 0.516-0.532 |
| incongruent accuracy | 0.453-0.472 | 0.449-0.478 | 0.496-0.528 | 0.490-0.505 |
| full-session routing flip | 0.637±0.047 | 0.625±0.051 | 0.688±0.045 | 0.575±0.071 |

Flat within noise across a 10x range of instruction repetition — no
monotonic trend in accuracy, congruency, or routing stability.
`cued_attractor`'s poor performance under cap-inversion isn't an
under-teaching artifact; the same near-chance, no-congruency-effect pattern
holds from 1 repeat to 10, so the low `practice_permutation_repeats` this
package had been left at wasn't masking a real effect above.

**Caveat:** `2cpr_slowW2` (the pre-existing 0.5 cap point) and
`irrelevant_leak_0.30` (`gated_simple_attractor`) weren't rerun this
session; the former is missing the new full-session metric, and the latter
is still at the old `practice_permutation_repeats=1` and isn't directly
comparable to the `baseline`/`cap_inverted` numbers above.

**Answers to section 14's "not yet checked" list:** whole-session stability
count — done (100% flip under the old config, 11.9% under `slowW3`, matches
section 13's "0/96" almost exactly). Finer cap sweep — done: accuracy
plateaus by around max_fast_weight≈0.3, while routing stability keeps
improving down to at least 0.1. Cross-package cap inversion — done:
gating-specific, not a general fix. **Still open:** re-tuning
`gating_to_feature_gain`/`gating_to_relevant_feature_gain` under
slow-dominant W; correlating `slowW3`'s lower-accuracy seeds with their
routing-flip rate.
