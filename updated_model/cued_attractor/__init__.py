"""Readable tools for the cued task-switching plastic-attractor model.

The shortest complete workflow is::

    result = run_baseline(seed=0)
    summary = summarize_behavior(result.trials)

This package keeps published_model's task / model / experiment / analysis split
and adds the cued-switching paradigm (cue units per rule, up-front practice
blocks, and real blocks where the rule is signalled by a cue) on top.
"""

from .analysis import (
    BehavioralSummary,
    Contrast,
    amplifying_eigenvalue_mean_by_kind,
    block_kinds,
    conjunction_routing_drift_by_kind,
    conjunction_routing_flip_rate_by_block,
    eigenvalue_magnitudes_by_kind,
    incongruence_contrast_by_kind,
    no_response_rate_by_block,
    performance_by_block,
    practice_learning_curve,
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
    cues_for_task,
    is_congruent,
)

__all__ = [
    'ALL_STIMULI',
    'PRACTICE_TASKS',
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
    'conjunction_routing_drift_by_kind',
    'conjunction_routing_flip_rate_by_block',
    'correct_response',
    'cues_for_task',
    'eigenvalue_magnitudes_by_kind',
    'incongruence_contrast_by_kind',
    'is_congruent',
    'no_response_rate_by_block',
    'performance_by_block',
    'practice_block_plan',
    'practice_learning_curve',
    'real_block_plan',
    'run_baseline',
    'run_epoch',
    'run_switching_experiment',
    'summarize_behavior',
    'switch_contrast_by_kind',
    'switching_practice_block_plan',
]
