"""Readable tools for the no-cue, oracle-suppression plastic-attractor model.

The shortest complete workflow is::

    result = run_baseline(seed=0)
    summary = summarize_behavior(result.trials)

This package keeps published_model's task / model / experiment / analysis
split. Unlike cued_attractor/gated_attractor, there are no cue units and no
gating units: the trial harness itself drives only the task-relevant feature
on every trial, leaving the irrelevant feature undriven by default
(ModelParameters.irrelevant_feature_drive) so it settles near baseline
through ordinary recurrent dynamics alone. This is an idealized,
oracle-suppression control at that default: if it achieves clean, high
accuracy, the core attractor's capacity/dynamics aren't the bottleneck in
gated_attractor, and the problem lives in the (learned, imperfect)
suppression mechanism instead.
"""

from .analysis import (
    ActivityLevels,
    BehavioralSummary,
    Contrast,
    amplifying_eigenvalue_mean_by_kind,
    block_kinds,
    colour_shape_row_norms_by_block,
    eigenvalue_magnitudes_by_kind,
    incongruence_contrast_by_kind,
    no_response_rate_by_block,
    performance_by_block,
    practice_learning_curve,
    relevant_irrelevant_activity_by_kind,
    summarize_behavior,
    switch_contrast_by_kind,
)
from .experiment import (
    PRACTICE_TASKS,
    ConjunctionClamp,
    EpochProtocol,
    ExperimentResult,
    NetworkTrajectory,
    PlannedTrial,
    SwitchingExperimentConfig,
    TimeWindow,
    TransitionType,
    TrialResult,
    author_tms_pulse,
    practice_block_plan,
    real_block_plan,
    run_baseline,
    run_epoch,
    run_switching_experiment,
    switching_practice_block_plan,
)
from .model import (
    ModelParameters,
    NetworkState,
    PlasticAttractor,
)
from .task import (
    ALL_STIMULI,
    Feature,
    Stimulus,
    Task,
    Vocabulary,
    build_vocabulary,
    correct_response,
    is_congruent,
)

__all__ = [
    'ALL_STIMULI',
    'PRACTICE_TASKS',
    'ActivityLevels',
    'BehavioralSummary',
    'ConjunctionClamp',
    'Contrast',
    'EpochProtocol',
    'ExperimentResult',
    'Feature',
    'ModelParameters',
    'NetworkState',
    'NetworkTrajectory',
    'PlannedTrial',
    'PlasticAttractor',
    'Stimulus',
    'SwitchingExperimentConfig',
    'Task',
    'TimeWindow',
    'TransitionType',
    'TrialResult',
    'Vocabulary',
    'amplifying_eigenvalue_mean_by_kind',
    'author_tms_pulse',
    'block_kinds',
    'build_vocabulary',
    'colour_shape_row_norms_by_block',
    'correct_response',
    'eigenvalue_magnitudes_by_kind',
    'incongruence_contrast_by_kind',
    'is_congruent',
    'no_response_rate_by_block',
    'performance_by_block',
    'practice_block_plan',
    'practice_learning_curve',
    'real_block_plan',
    'relevant_irrelevant_activity_by_kind',
    'run_baseline',
    'run_epoch',
    'run_switching_experiment',
    'summarize_behavior',
    'switch_contrast_by_kind',
    'switching_practice_block_plan',
]
