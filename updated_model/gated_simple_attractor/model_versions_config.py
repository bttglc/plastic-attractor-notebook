"""Named ModelParameters sets for model-version comparison, imported by launcher.py.

Add entries here to compare against the shared Whyte model parameters.
"""

from gated_simple_attractor import ModelParameters

# number of conjunction units, common to every version below unless a
# version overrides it
number_of_conjunction_units = 4

model_versions = {
    # same gains as cued_attractor's whyte_params_2cpr / gated_attractor's
    # 2cpr_gating_units, minus every cue/gate-specific field -- neither
    # exists in this package.
    'baseline': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
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
