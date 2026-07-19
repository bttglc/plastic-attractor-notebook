# gated_attractor model outline

Reference for the package in this folder. Written after fixing gate
non-selectivity and investigating (unsuccessfully so far) why gating still
doesn't clearly help behaviour despite correct structural selectivity. See
'open questions' at the end for the current state of that investigation.

## 1. Lineage

Extends Whyte et al. 2025, 'A plastic attractor model of flexible rule-based
selective attention' (bioRxiv 2025.09.07.674747; in Zotero as item 7L2R5LYQ),
itself built on the 'plastic attractor' architecture from Manohar et al. 2019.
`model.py`'s core equations are a near-verbatim copy of the published model
(`published_model/plastic_attractor/model.py`); this package adds a cued
task-switching paradigm on top: instead of re-teaching a fixed rule every
block, the network is taught once (two instruction blocks), then runs real
blocks where the rule varies trial-by-trial and is signalled only by a cue.

## 2. Neural populations

Three populations, defined in `model.py`'s `PlasticAttractor`:

- **Feature units** (10 by default, `num_cues_per_rule=2`): green, blue,
  square, circle, 2 colour-rule cues, 2 shape-rule cues, action1, action2
  (index layout built by `task.py`'s `build_vocabulary`). Colour/shape/cue/
  action each form a competing group with within-group lateral inhibition
  (`feature_lateral_weight=-0.28`) and self-excitation
  (`feature_self_weight=0.73`).
- **Conjunction units** (4 by default): globally competing (winner-take-all,
  `conjunction_lateral_weight=-0.45`, `conjunction_self_weight=1.00`),
  recurrently coupled to feature units via the plastic weight matrix W.
- **Gating units** (0 or 2, one per rule; `number_of_gating_units`): optional
  feedforward inhibitory interneurons, driven by the cue, meant to learn to
  suppress the task-irrelevant colour/shape dimension. 0 disables them
  entirely and reproduces the published (Whyte et al.) dynamics bit-for-bit.
  Also globally competing (`gating_lateral_weight=-0.45`,
  `gating_self_weight=1.6` -- see 'gate persistence' below).

## 3. Task / vocabulary (`task.py`)

- `Feature`: GREEN, BLUE, SQUARE, CIRCLE (fixed indices 0-3).
- `Task`: COLOR, SHAPE -- the rule, i.e. which dimension determines the
  correct response.
- `Stimulus`: one (colour, shape) pair; `relevant_feature(task)` /
  `irrelevant_feature(task)` pick the dimension that matters / doesn't.
- `Vocabulary`: lays out cue and action indices after the 4 fixed
  colour/shape ones. `gate_target_indices` hardcodes, per gate (colour gate
  first, shape gate second, matching `PRACTICE_TASKS`' order), which feature
  rows that gate may ever suppress -- its task's irrelevant pair, never its
  own. This is the fix from this session: previously gate output weights had
  no way to learn this distinction on their own (see 'fixed: gate
  non-selectivity' below).
- Stimuli are congruent when colour and shape indicate the same response
  (`is_congruent`) -- green+square and blue+circle by construction.

## 4. Dynamics (`model.py::PlasticAttractor.step`)

One time step, given `external_input` (length = number of feature units):

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

`centered_x = x - baseline_activity` (0.175) throughout -- every recurrent/
plastic term is covariance-style, relative to a resting baseline, not raw
activity. All three populations update synchronously from the same prior
state. `clip(., 0, 1)` (`_bounded_activity`) is what makes suppression
provably unable to act while a feature is externally floored at 1.0 -- see
'open questions'.

**Gate persistence.** `gating_self_weight=1.6 > 1` gives the 2-gate
winner-vs-loser mode an eigenvalue > 1: once a brief cue-driven push (during
`stimulus_window`) tips one gate ahead, its own recurrence keeps it latched
near ceiling for the rest of the trial without needing the cue to stay on.
`_next_gating_activity` only reads cue-unit activity while
`cue_signal_active` (scoped to `stimulus_window`), not continuously --
otherwise bottom-up conjunction feedback can partially re-light the wrong
cue mid-trial and drag an already-settled gate off its decision.

## 5. Learning rules

**Main W (feature <-> conjunction), `_update_plastic_weights`:** raw,
per-step, unsupervised Hebbian covariance rule, run every learning step with
no reward gating, exactly as published:
```
change = outer(centered_features, centered_conjunctions)
change[cue_indices] = 0        # cues never bind into W; they only reach the
                                # network via the gating pathway
fast_weights = clip(fast_weights + fast_learning_rate * change, 0, 1.0)     # gamma1=0.02
slow_weights = clip(slow_weights + slow_learning_rate * change, 0, 0.2)     # gamma2=0.0002
W = fast_weights + slow_weights
```
Fast/slow split (and the analogous split for gating weights below) is
Whyte et al.'s addition over the base Manohar et al. model: without a slow,
narrowly-bounded component anchoring which conjunction unit represents which
mapping, conjunction-unit selectivity drifts block to block and the
congruency effect disappears (their Supplementary finding, reproduced by our
`_combine_weights` == `fast_weight_blend*fast + slow_weight_blend*slow`).

**Gating weights (cue -> gate input, gate -> feature output):** a
three-factor / eligibility-trace rule (Fremaux & Gerstner 2016), separate
from W's rule above:
1. `_accumulate_gating_trace` builds a decayed running trace
   (`gating_trace_decay=0.98`) every step, but only while `cue_signal_active`
   (`stimulus_window`) -- outside that window the trace neither forms nor
   decays.
2. The trial's dynamics run under *fixed* gate weights throughout; nothing
   is applied until the outcome is known.
3. `consolidate_gating_trace(reward)` (called once per trial, every trial,
   from `experiment.py::_run_trial`) applies the accumulated trace scaled by
   `+1` (gate picked the correct rule) or `-1` (it didn't) -- the
   bidirectional dopamine-gated LTP/LTD pattern reported at corticostriatal
   synapses (Shen et al. 2008). Reward is whether *the gate itself* won the
   correct side (`gate_winner_matches_task`), not overall trial correctness
   -- using overall correctness let unrelated W/congruency errors erode an
   already-correct gate mapping (this was an earlier, discarded design).

Output weights are additionally masked (`_gating_output_mask`, built from
`gate_target_indices`) at both initialization and every trace-accumulation
step, so a gate's own dimension can structurally never accumulate a nonzero
trace or start with nonzero weight -- this is the fix described below.

## 6. Trial timing (`experiment.py::EpochProtocol`)

400 steps per trial by default:
- `[0, 51)`: ISI, input = -1 on every feature.
- `stimulus_window = [51, 101)`: stimulus (colour+shape) and cue presented
  together, both at 1.0.
- `teaching_window = [50, 351)` (instruction trials only): action units
  forced to +1 (correct) / -1 (incorrect), overriding whatever the network
  would otherwise settle on -- starts one step before `stimulus_window` and
  extends well past it so the forced signal survives the Hebbian update
  running every step.
- `response_window = [101, 351)`: zero input: the network settles into an
  attractor from recurrence and W-feedback alone. `response_search_start=110`
  is where `_measure_response` starts scanning for a 98%-of-peak crossing on
  the action units (the global-peak-then-threshold rule from the flat
  original script).
- `[351, 400)`: ISI again.

Stimulus and cue are *always* presented together, at full strength,
regardless of gating -- teaching no longer overrides the irrelevant
colour/shape pair with an artificial neutral value (removed this session);
suppressing it is entirely the gate's job now, on instruction and real
trials alike.

## 7. Experiment structure (`run_switching_experiment`)

1. **Instruction** (`num_practice_blocks=2`, one per rule by default): every
   (cue x stimulus) combination, `practice_permutation_repeats` times,
   response forced via `teaching_window`, gate forced via `_gate_drive_
   schedule` (`+1` on the correct gate, `-1` on the other, held only during
   `stimulus_window`).
2. **Performance practice** (`include_performance_practice=True`): one block
   per rule, `num_trials` each, no forcing -- the network's own, unforced
   behaviour, with learning still on.
3. **Real / switching blocks**: one per entry in `switch_probs`, rule varies
   trial-to-trial at that switch probability, signalled only by the cue.
   Learning stays on throughout (`learn_during_trials`), so weights keep
   evolving during real trials too, not just instruction.

`model_versions_config.py` holds named `ModelParameters` presets;
`2cpr_gating_units` is the only one that turns gating on
(`number_of_gating_units=2`, `gating_to_feature_gain=0.4`,
`gating_self_weight=1.6`, etc. -- tuned, per its own comments, against the
*pre-fix* non-selective gating mechanism, and not yet re-validated against
the now-selective one).

## 8. Diagnostics (`analysis.py`, `launcher.py`)

`launcher.py` runs a model version across `n_seeds` (parallelized via
`ProcessPoolExecutor`), reduces trials with `analysis.py`'s functions, and
writes `output/<version>/simulation_data.npz` plus figures. Beyond the
original accuracy/RT/congruency/switch-cost summaries, it now also collects
(added this session, to keep investigating without hand-rolled probes):
`gate_accuracy_by_block` (did the gate pick the right rule, per block),
`no_response_rate_by_block` (network never reached threshold, vs. reaching
the wrong one), `colour_shape_row_norms_by_block` (W row-norm evolution for
green/blue/square/circle), and `relevant_irrelevant_activity_by_kind`
(settled feature activity split by whether the current rule cares about it).

## 9. Fixed this session: gate non-selectivity

**Symptom:** `test_practice_teaches_above_chance_behaviour` failed
(0.458 accuracy, below chance) before any of this session's changes.

**Root cause:** both dimensions are simultaneously active with the correct
gate throughout every `stimulus_window`, regardless of which one the rule
cares about -- the eligibility trace had no signal to prefer suppressing one
over the other, so gate output weights saturated near ceiling on *both*
dimensions symmetrically instead of selectively on the irrelevant one.

**Fix:** hardcoded structural masking (`gate_target_indices` +
`_gating_output_mask`), not learned -- see section 5. Verified: gate output
weights are now exactly selective (own-dimension weight = 0.0 always,
confirmed across seeds); gate winner matches the true task 100% of the time
in every check run so far; the previously-failing test now passes.

## 10. Open questions -- selectivity fixed, behaviour still not clearly better

Despite (9) being verified correct, gated real-block accuracy is not
reliably above the pre-fix baseline, and multiple attempts this session to
go further haven't resolved it either:

**Confirmed structural finding:** gate suppression is arithmetically
incapable of acting while the stimulus is on screen. During
`stimulus_window`, the irrelevant feature is externally floored at 1.0;
maximum possible suppression at current tuning
(`gating_to_feature_gain=0.4` x max combined gate weight 1.2 x max gate
deviation 0.825 ~= 0.396) is under half of what's needed to pull the
`clip(., 0, 1)`-bounded sum below 1.0. Confirmed empirically, not just
arithmetically: irrelevant-feature minimum activity *during*
`stimulus_window` measured exactly 1.000 in every trial checked, every
seed, regardless of how well-trained the gate is.

**Consequence:** `_update_plastic_weights` runs every step, including all of
`stimulus_window`, so every trial (instruction and real alike) Hebbian-binds
the fully-active irrelevant feature to whatever conjunction unit is active
at that moment -- confirmed in the actual W structure: some conjunction
units end up with near-equal weights on both options of a competing pair
(e.g. green and blue both high on the same unit), unable to discriminate
within that dimension at all.

**Tried, reverted, then merged (see section 11):** excluding the
currently-winning gate's target rows from W's Hebbian update during
`stimulus_window` + a ~10-step buffer (letting suppression's own ~10-step
time constant catch up before learning resumes). Verified this measurably
improved raw weight-level within-dimension discrimination *relative to
blunter pause variants* (colour/shape gaps up to ~0.9-1.0 in several seeds,
vs. near-zero everywhere before) -- but real-block accuracy did not improve
(~0.51 vs ~0.495 baseline, 12 seeds, `practice_permutation_repeats=1`), and
`test_practice_teaches_above_chance_behaviour` regressed. Two blunter
variants (pausing the whole update; pausing the whole suppressible block
regardless of which gate was winning) were tried first and were worse still.
Section 11 found that this comparison -- against blunter pause variants --
was the wrong baseline: checked properly against gating with *no* pause at
all, this same mechanism left weight selectivity worse, not better, until
combined with the gain re-tuning described there.

**Why the disconnect might exist -- not yet distinguished:**
- *Measurement gap in how "improvement" was checked.* Verification so far
  measured the *best* (max) discrimination gap across the 4 conjunction
  units per seed. A network only behaves correctly if enough units are
  simultaneously well-differentiated -- Whyte et al.'s own criterion is
  >=2 amplifying eigenvalues of WW^T, one per genuinely distinct S-R
  mapping (congruent pairs can legitimately share a unit; that's not a
  failure). "Best gap improved" is compatible with the *other* units still
  being collapsed or duplicated, which would leave whole stimulus/rule
  combinations unsolvable even as the aggregate metric looks better. Should
  check *all four* units' gaps, or amplifying-eigenvalue count by block
  (already collected), or per-task-type accuracy, not just the max.
- *Selectivity in W isn't the same as correct competition outcomes.* A unit
  can have a large weight gap and still not be the one that wins
  winner-take-all on the trials that need it, if its recurrent/bottom-up
  drive doesn't dominate the competition. Discrimination in weight space is
  necessary but not obviously sufficient.
- *Relevant-feature settled activity is unexpectedly modest even outside
  gating* (~0.3-0.5 typical, not near ceiling, in both gating and no-gating
  runs) -- worth checking whether this is a general property of the current
  parameter regime (4 conjunction units, current gains) rather than
  something gating specifically worsens, since a weak/noisy settled signal
  would make the 98%-of-peak response measure sensitive to small,
  non-meaningful fluctuations regardless of which fix is tried.
- *Capacity is tight by construction.* 4 conjunction units for 2 rules x 2
  responses (with only congruent pairs able to legitimately share a unit)
  leaves very little redundancy -- Whyte et al.'s own >=2-amplifying-
  eigenvalue minimum is a bare floor, not a comfortable margin, so small
  perturbations from any fix could easily leave some trial types
  under-served even while aggregate numbers move in the right direction.

**Candidate task redesign (not yet tried):** present the cue *before*
stimulus onset (rather than simultaneously, as now) so gate suppression has
already engaged and is subtracting from the moment the stimulus arrives,
rather than only after fighting down from an already-floored 1.0. This
targets the confirmed root cause directly and more fundamentally than
after-the-fact Hebbian masking does -- suppression would be preventing the
irrelevant feature from ever reaching ceiling, not un-learning the
consequences of it having done so. It's also a well-established real
experimental manipulation in the human task-switching literature (cue-
stimulus interval / CSI, e.g. Rogers & Monsell 1995, Meiran 1996), not an
artificial addition. Bigger structural change than a parameter tweak
(`_trial_epoch`'s input schedule would need cue and stimulus on separate
windows, and `stimulus_window`/`teaching_window`/gate-drive timing would
need to follow) -- worth trying if the measurement-gap explanation above
doesn't turn out to be the whole story.

## 11. This session: plasticity pause merged, re-tuned with a relevant-
feature enhancement pathway, and a corrected measurement practice

Follow-up to (10). Four things changed, in this order.

**Plasticity pause merged into the active package.** `wip_plasticity_
pause_weight_mask/` (model.py's `pause_weight_learning` param and
`_update_plastic_weights`' gate-target-aware row exclusion; experiment.py's
`plasticity_pause_buffer_steps` / `plasticity_pause_window` and `run_epoch`
wiring) is now just how `_update_plastic_weights` and `run_epoch` work --
the snapshot folder is deleted, its content was byte-identical to the merge
(diffed to confirm before deleting). Still only takes effect when gating is
on; gating off stays bit-identical to Whyte et al.

**Corrected measurement practice -- the actual point of (10)'s "measurement
gap" worry.** `analysis.py` gained `conjunction_unit_discrimination`
(per-unit colour/shape weight gap, shape `(num_conjunction_units, 2)`, not
collapsed to a max) and `accuracy_by_task` / `accuracy_by_task_by_kind`
(colour-rule vs shape-rule accuracy, not just pooled). Applying these,
rather than the old max-gap check, to a straight pause-vs-no-pause
comparison (n=20 seeds, `2cpr_gating_units`-style parameters) overturned
(10)'s own framing: the pause's "improved... within-dimension
discrimination" claim was only ever true *relative to blunter pause
variants*. Against gating with no pause at all, the same pause left weight
selectivity **worse**: mean well-separated conjunction units (both colour
and shape gap > 0.3) fell from 1.20/4 (gating, no pause) to 0.10/4 (gating
+ pause, both at the old `gating_to_feature_gain=0.4`) -- pooled accuracy
stayed flat either way (~0.51). **Going forward, check all
`number_of_conjunction_units` units (or the full eigenvalue spectrum /
per-task accuracy split), never a single best/max/pooled number, when
judging whether a change helped selectivity or behaviour** -- this is now
standard practice for this investigation, not a one-off fix.

**New mechanism: fixed, multiplicative relevant-feature enhancement
(`gating_to_relevant_feature_gain`, model.py).** Tests whether directly
boosting the relevant dimension helps, on top of suppressing the irrelevant
one -- motivated by (10)'s finding that suppression alone is arithmetically
capped against the stimulus's external floor. Structurally the mirror image
of the existing suppression pathway (`task.py`'s new
`gate_relevant_target_indices`, a fixed 0/1 mask built once in
`PlasticAttractor.__init__`, never learned). **First version (flat
additive, matching gating_inhibition's structure) was actively harmful at
every gain tried**: it adds the same constant to *both* members of the
relevant pair (e.g. green and blue alike, since a gate only knows the
relevant *dimension*, not which member is actually shown), which inflates
the absent member as much as the presented one and collapses within-pair
discrimination -- confirmed empirically (well-separated units fell to ~0
the moment any additive gain > 0 was added, regardless of inhibition
strength). **Fixed to multiplicative**: the enhancement term is
`gain * gate_deviation * centered_features`, i.e. scaled by the feature's
own current deviation from baseline, so it amplifies whichever member is
already driven up by the real stimulus (the absent member's near-zero
centered activity gives it near-zero enhancement) instead of blurring the
pair together. Default `0.0` -- exact no-op, bit-identical to before this
existed.

**Re-tuned via grid + fine sweep (parallelised, `ProcessPoolExecutor`,
scratch scripts, not committed).** Swept `gating_to_feature_gain` x
`gating_to_relevant_feature_gain` (coarse 4x4 grid at n=8 seeds, then a
finer 5x4 grid at n=10 around the best cell). Findings: raising inhibition
alone (enhancement=0) monotonically raised well-separated-unit count
(0.12 -> 0.25 -> 0.50 -> 0.50 of 4, gains 0.4/0.7/1.0/1.3) without moving
accuracy; adding the (corrected, multiplicative) enhancement on top of a
*moderate* inhibition raise was what actually moved accuracy, and not
monotonically -- `gating_to_feature_gain=0.8` was reliably *worse* than
0.7 despite comparable or better structural metrics, so this is a
sweet-spot, not "more is better" in either gain. **Chosen and validated at
n=20 seeds**: `gating_to_feature_gain=0.7`, `gating_to_relevant_feature_
gain=0.6`.

| | gating off | gating+pause, old gains (0.4/0.0) | gating+pause, tuned (0.7/0.6) |
|---|---|---|---|
| real-block accuracy | 0.506 +/- 0.018 | 0.510 +/- 0.006 | **0.565 +/- 0.028** |
| colour / shape accuracy | 0.516 / 0.498 | 0.524 / 0.496 | 0.585 / 0.541 |
| well-separated units (/4) | 1.20 | 0.10 | **2.00** |
| amplifying eigenvalues (last block) | 2.60 | 2.15 | 2.55 |

The tuned setting is the first configuration this whole investigation (all
of sections 9-11) has found where accuracy, colour/shape balance, and
per-unit structural selectivity all move together in the right direction
at once, at the seed count this project trusts. `model_versions_config.py`'s
`2cpr_gating_units` now uses these values (with the sweep numbers recorded
inline); `test_practice_teaches_above_chance_behaviour` now sets them
explicitly rather than relying on `ModelParameters`' conservative class
defaults (which stay 0.4/0.0 -- backward-compatible, not re-tuned) and its
own scenario goes from 0.417 (failing) to 0.792 with them, seed 0.

**Not resolved / worth a future look:** why `gating_to_feature_gain=0.8`
specifically breaks accuracy while 0.7 and 0.9 both work reasonably well
wasn't chased down -- the sweep found the effect, not the mechanism. Colour
accuracy still edges out shape accuracy in most conditions tried
(0.585 vs 0.541 at the tuned setting); worth watching if it grows into a
genuine asymmetry rather than sampling noise. Full launcher figures (20
seeds, all diagnostics) haven't been regenerated with the new tuned values
yet -- the numbers above come from focused scratch comparisons, not
`launcher.py`'s output directory.
