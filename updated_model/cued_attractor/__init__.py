"""Readable tools for the cued task-switching plastic-attractor model.

The shortest complete workflow is::

    result = run_baseline(seed=0)
    summary = summarize_behavior(result.trials)

This package keeps published_model's task / model / experiment / analysis split
and adds the cued-switching paradigm (four cue units, up-front practice blocks,
and real blocks where the rule is signalled by a cue) on top.
"""

from .analysis import (
    BehavioralSummary,
    Contrast,
    amplifying_eigenvalue_mean_by_kind,
    block_kinds,
    eigenvalue_magnitudes_by_kind,
    incongruence_contrast_by_kind,
    performance_by_block,
    practice_learning_curve,
    summarize_behavior,
    switch_contrast_by_kind,
)
from .experiment import (
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
)
from .model import (
    ModelParameters,
    NetworkState,
    PlasticAttractor,
)
from .task import (
    ALL_STIMULI,
    Cue,
    Feature,
    Stimulus,
    Task,
    correct_response,
    cues_for_task,
    is_congruent,
)

__all__ = [
    'ALL_STIMULI',
    'BehavioralSummary',
    'ConjunctionClamp',
    'Contrast',
    'Cue',
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
    'amplifying_eigenvalue_mean_by_kind',
    'author_tms_pulse',
    'block_kinds',
    'correct_response',
    'cues_for_task',
    'eigenvalue_magnitudes_by_kind',
    'incongruence_contrast_by_kind',
    'is_congruent',
    'performance_by_block',
    'practice_block_plan',
    'practice_learning_curve',
    'real_block_plan',
    'run_baseline',
    'run_epoch',
    'run_switching_experiment',
    'summarize_behavior',
    'switch_contrast_by_kind',
]
