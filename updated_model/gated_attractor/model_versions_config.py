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
        # raised from the default 0.08: needs to outcompete the feature's
        # own recurrent settling (feature_self_weight=0.73) over the
        # ~250-step coasting period after the cue flash ends, not just
        # nudge it. 0.08 (matched to conjunction_to_feature_gain) was tried
        # first and left suppression unmeasurable
        gating_to_feature_gain=0.4,
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
}
