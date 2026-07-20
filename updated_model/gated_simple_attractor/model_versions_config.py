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
    # same as baseline but with a partial (non-oracle) irrelevant-feature
    # drive, so congruency has a route into a trial's dynamics. 0.30 chosen
    # from a coarse scratch sweep (n=6 then n=8 seeds, 0.15-0.7): the only
    # value that replicated a real congruency effect across two independent
    # seed batches (+15.4 and +17.8 accuracy points), landing close to the
    # one validated "real effect" precedent in this project family (13-18
    # points at 61.9% accuracy, project/whyte_model/PlasticAttractorCode's
    # cued extension). Not yet confirmed at n=20 -- see output/ for whichever
    # run is current.
    'irrelevant_leak_0.30': ModelParameters(
        number_of_conjunction_units=number_of_conjunction_units,
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=1.0, maximum_slow_weight=0.2,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
        irrelevant_feature_drive=0.30,
    ),
}
