"""Readable tools for running and extending the plastic-attractor model.

The shortest complete workflow is::

    result = run_baseline(seed=0)
    summary = summarize_behavior(result.trials)
"""

from .analysis import BehavioralSummary, summarize_behavior
from .experiment import (
    BlockedExperimentConfig,
    ConjunctionClamp,
    EpochProtocol,
    ExperimentResult,
    NetworkTrajectory,
    TimeWindow,
    TrialResult,
    author_tms_pulse,
    run_baseline,
    run_blocked_experiment,
    run_epoch,
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
    correct_response,
    is_congruent,
)

__all__ = [
    "ALL_STIMULI",
    "BehavioralSummary",
    "BlockedExperimentConfig",
    "ConjunctionClamp",
    "EpochProtocol",
    "ExperimentResult",
    "Feature",
    "ModelParameters",
    "NetworkState",
    "NetworkTrajectory",
    "PlasticAttractor",
    "Stimulus",
    "Task",
    "TimeWindow",
    "TrialResult",
    "author_tms_pulse",
    "correct_response",
    "is_congruent",
    "run_baseline",
    "run_blocked_experiment",
    "run_epoch",
    "summarize_behavior",
]
