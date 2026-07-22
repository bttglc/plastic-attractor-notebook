"""Named ModelParameters sets for model-version comparison, imported by launcher.py.

Add entries here to compare against the original Whyte model parameters.
"""

from cued_attractor import ModelParameters

# number of cues per rule and of conjunction units, common to every version
# below unless a version overrides them
num_cues_per_rule = 2
number_of_conjunction_units = 4

# _build_conjunction_connections() returns self_weight*eye(N) + lateral_weight
# * ones(N, N): the all-ones term adds lateral_weight to every entry,
# including the diagonal, so the row sum is self_weight + N*lateral_weight.
# whyte_params (-0.45, N=4) tunes that sum to 1.0 + 4*(-0.45) = -0.8. To keep
# the same row sum (and hence the same population-mean-mode eigenvalue) at
# other conjunction-unit counts, lateral_weight must scale as baseline_N / N.
_baseline_conjunction_units = 4
_baseline_conjunction_lateral_weight = -0.45


def _rescaled_conjunction_lateral_weight(n_conjunction_units):
    return (
        _baseline_conjunction_lateral_weight
        * _baseline_conjunction_units
        / n_conjunction_units
    )

model_versions = {
    'whyte_params_2cpr': ModelParameters(
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
    ),
    '2cpr_Wslow_cap_high': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        num_cues_per_rule=num_cues_per_rule,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=1.0, maximum_slow_weight=0.7,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
    ),
    '2cpr_Wslow_cap_high2': ModelParameters(
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
    ),
    # V1: one cue per rule, baseline (whyte_params) conjunction-unit count
    'V1_1cpr_4conj': ModelParameters(
        number_of_conjunction_units=4,
        num_cues_per_rule=1,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=1.0, maximum_slow_weight=0.2,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
    ),
    # V1b: one cue per rule, conjunction units stepped up from V1's 4 to 6
    'V1b_1cpr_6conj': ModelParameters(
        number_of_conjunction_units=6,
        num_cues_per_rule=1,
        conjunction_lateral_weight=_rescaled_conjunction_lateral_weight(6),
        conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=1.0, maximum_slow_weight=0.2,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
    ),
    # V2: back to two cues per rule, conjunction units stepped up from V1's 4 to 6
    'V2_2cpr_6conj': ModelParameters(
        number_of_conjunction_units=6,
        num_cues_per_rule=2,
        conjunction_lateral_weight=_rescaled_conjunction_lateral_weight(6),
        conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=1.0, maximum_slow_weight=0.2,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
    ),
    # V3: two cues per rule, conjunction units stepped up further to 8
    'V3_2cpr_8conj': ModelParameters(
        number_of_conjunction_units=8,
        num_cues_per_rule=2,
        conjunction_lateral_weight=_rescaled_conjunction_lateral_weight(8),
        conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=1.0, maximum_slow_weight=0.2,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
    ),
}
