"""Named ModelParameters sets for model-version comparison, imported by launcher.py.

Add entries here to compare against the original Whyte model parameters.
"""

from gated_attractor import ModelParameters

# number of cues per rule and of conjunction units, common to every version
# below unless a version overrides them
num_cues_per_rule = 2
number_of_conjunction_units = 4

model_versions = {
    # adds 2 gating units (one per rule) on top of the whyte_params_2cpr
    # base: self-sustaining inhibitory interneurons, driven by the cue, that
    # learn to suppress the task-irrelevant colour/shape pair, on instruction
    # and real trials alike (both present the stimulus the same way; only the
    # cue says which rule is active). comments below cover
    # only the gating_* fields and why each one's value was chosen; the
    # mechanism itself (why the gate reads the cue only during
    # stimulus_window, why consolidation reward is the gate's own winner
    # rather than overall response correctness) lives in model.py/
    # experiment.py next to the code it explains.
    '2cpr_gating_units': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=1.0, maximum_slow_weight=0.2,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        # one gate per rule (colour, shape); 0 disables gating entirely and
        # reproduces whyte_params_2cpr bit-for-bit
        number_of_gating_units=2,
        # cue -> gate input gain, left at default: persistence through the
        # trial comes from gating_self_weight below, not from this
        cue_to_gating_gain=1.0,
        # re-tuned this session (was 0.4, tuned only against the pre-fix
        # non-selective gating mechanism -- see model_outline.md section 11):
        # with the gate-target-aware plasticity pause now active (excludes
        # the currently-winning gate's target rows from W's Hebbian update
        # during stimulus_window + a buffer), 0.4 left per-unit weight
        # discrimination *worse* than gating off entirely (mean 0.10 of 4
        # conjunction units well-separated, vs 1.20 with gating off, n=20
        # seeds). A grid sweep over this gain and gating_to_relevant_
        # feature_gain below found 0.7 the best-supported value: raising it
        # further (0.8+) reliably got worse again despite higher structural
        # selectivity, so this isn't "more suppression is always better"
        gating_to_feature_gain=0.7,
        # new this session (see model.py's gating_to_relevant_feature_gain):
        # fixed, multiplicative excitatory pathway from a gate onto its own
        # (relevant) dimension, on top of the suppression above. 0.6 chosen
        # by the same sweep; validated at n=20 seeds together with the 0.7
        # inhibition gain above: real-block accuracy 0.565 (vs 0.506 with
        # gating on but these two gains at their old/default values, and
        # 0.510 with gating on and the plasticity pause but pre-sweep
        # gains), and well-separated conjunction units 2.00/4 on average
        # (vs 1.20 with gating off and 0.10 with the pause at old gains)
        gating_to_relevant_feature_gain=0.6,
        # fast/slow split, learning rates, weight caps and blends left at
        # the same values used for the main plastic weights above: fast
        # tracks each trial's reward-gated update, slow anchors the
        # instruction-taught mapping against real-block drift
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        # eligibility trace decay; the trace only accumulates during
        # stimulus_window (~50 steps, see experiment.py), so ~0.98 already
        # covers that whole window without needing to change
        gating_trace_decay=0.98,
        # raised from the default 1.00: with 2 gates the winner-vs-loser
        # difference mode has eigenvalue == gating_self_weight exactly, so
        # >1 makes it a growing map that latches at the clipped ceiling and
        # holds there via pure self-recurrence once the brief cue flash
        # ends, rather than decaying back toward baseline. this is what
        # lets one gate stay the correct winner for the rest of the
        # ~400-step trial from only a ~50-step external cue; 1.00 (matching
        # the conjunction units) was tried first and decayed to a
        # submaximal, easily-disturbed steady state instead of latching
        gating_self_weight=1.6,
        # unchanged: same winner-take-all lateral inhibition as the
        # conjunction units
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.7, maximum_slow_weight=0.7,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.7,
        gating_to_relevant_feature_gain=0.6,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW2': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.5, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.7,
        gating_to_relevant_feature_gain=0.6,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW3': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.2, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.7,
        gating_to_relevant_feature_gain=0.6,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    # finer cap sweep between slowW2 (max_fast=0.5, acc 0.842) and slowW3
    # (max_fast=0.2, acc 0.881), max_slow_weight held at 1 throughout, to
    # locate where the accuracy gain from slow-dominant W plateaus (section
    # 14's "not yet checked": slowW3 vs slowW4 was already a plateau, not a
    # trend, at the high end of dominance -- this checks the low end)
    '2cpr_slowW2_cap01': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.1, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.7,
        gating_to_relevant_feature_gain=0.6,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW2_cap03': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.3, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.7,
        gating_to_relevant_feature_gain=0.6,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW2_cap04': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.4, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.7,
        gating_to_relevant_feature_gain=0.6,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    # gating-gain re-tuning under slow-dominant W (section 15's "still open"
    # item): gating_to_feature_gain/gating_to_relevant_feature_gain were
    # tuned to 0.7/0.6 in section 11 against 2cpr_gating_units' old
    # fast-dominant W (max_fast=1.0/max_slow=0.2). All 8 entries below are
    # 2cpr_slowW3 (max_fast=0.2/max_slow=1) with gain pairs from a 3x3 grid
    # around that optimum; the (0.7, 0.6) centre point is slowW3 itself, not
    # rerun here.
    '2cpr_slowW3_g05_r04': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.2, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.5,
        gating_to_relevant_feature_gain=0.4,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW3_g05_r06': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.2, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.5,
        gating_to_relevant_feature_gain=0.6,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW3_g05_r08': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.2, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.5,
        gating_to_relevant_feature_gain=0.8,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW3_g07_r04': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.2, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.7,
        gating_to_relevant_feature_gain=0.4,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW3_g07_r08': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.2, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.7,
        gating_to_relevant_feature_gain=0.8,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW3_g09_r04': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.2, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.9,
        gating_to_relevant_feature_gain=0.4,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW3_g09_r06': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.2, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.9,
        gating_to_relevant_feature_gain=0.6,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    '2cpr_slowW3_g09_r08': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.2, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.9,
        gating_to_relevant_feature_gain=0.8,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
    # Worse than 2cpr_slowW3
     '2cpr_slowW4': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=0.2, maximum_slow_weight=1,
        fast_weight_blend=1.0, slow_weight_blend=1.5,
        number_of_gating_units=2,
        cue_to_gating_gain=1.0,
        gating_to_feature_gain=0.7,
        gating_to_relevant_feature_gain=0.6,
        gating_fast_learning_rate=0.02, gating_slow_learning_rate=0.0002,
        gating_maximum_fast_weight=1.0, gating_maximum_slow_weight=0.2,
        gating_fast_weight_blend=1.0, gating_slow_weight_blend=1.0,
        gating_trace_decay=0.98,
        gating_self_weight=1.6,
        gating_lateral_weight=-0.45,
    ),
}
