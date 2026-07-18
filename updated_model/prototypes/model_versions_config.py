"""Named ModelParameters sets for model-version comparison, imported by launcher.py.

Add entries here to compare against the original Whyte model parameters.
"""

from cued_attractor import ModelParameters

# number of cues per rule and of conjunction units, common to every version
# below unless a version overrides them
num_cues_per_rule = 2
number_of_conjunction_units = 4

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
}
