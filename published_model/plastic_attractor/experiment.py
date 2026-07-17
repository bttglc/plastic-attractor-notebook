"""Timing, inputs, results, and the published blocked experiment.

This module translates experimental concepts into external inputs for the
network. It is the natural place to change the task design later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import NUMBER_OF_FEATURE_UNITS, ModelParameters, PlasticAttractor
from .task import (
    ALL_STIMULI,
    FEATURES_BY_TASK,
    IRRELEVANT_FEATURES_BY_TASK,
    RESPONSE_BY_FEATURE,
    RESPONSE_FEATURES,
    Feature,
    Stimulus,
    Task,
    correct_response,
    is_congruent,
)


# The experiment dataclasses keep related settings and results together.
@dataclass(frozen=True)
class TimeWindow:
    """A half-open interval: start is included and stop is excluded."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("A time window needs 0 <= start < stop")

    def contains(self, time_step: int) -> bool:
        return self.start <= time_step < self.stop

    @property
    def duration(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class EpochProtocol:
    """Timing used by the public implementation."""

    number_of_steps: int = 400
    instruction_window: TimeWindow = field(
        default_factory=lambda: TimeWindow(0, 251)
    )
    stimulus_window: TimeWindow = field(
        default_factory=lambda: TimeWindow(51, 101)
    )
    response_window: TimeWindow = field(
        default_factory=lambda: TimeWindow(101, 351)
    )
    response_search_start: int = 110
    repeats_per_stimulus: int = 3

    def __post_init__(self) -> None:
        if self.number_of_steps <= 0:
            raise ValueError("number_of_steps must be positive")
        for window in (
            self.instruction_window,
            self.stimulus_window,
            self.response_window,
        ):
            if window.stop > self.number_of_steps:
                raise ValueError("Protocol windows must fit inside the epoch")
        if not 0 <= self.response_search_start < self.number_of_steps:
            raise ValueError("response_search_start must lie inside the epoch")
        if self.repeats_per_stimulus <= 0:
            raise ValueError("repeats_per_stimulus must be positive")


@dataclass(frozen=True)
class ConjunctionClamp:
    """Force every conjunction unit to maximum activity in a time window."""

    window: TimeWindow

    @classmethod
    def between(cls, start: int, stop: int) -> ConjunctionClamp:
        return cls(TimeWindow(start, stop))

    @classmethod
    def ending_with_stimulus(
        cls,
        duration_in_steps: int,
        protocol: EpochProtocol | None = None,
    ) -> ConjunctionClamp:
        if duration_in_steps <= 0:
            raise ValueError("duration_in_steps must be positive")

        timing = protocol or EpochProtocol()
        stop = timing.stimulus_window.stop
        start = stop - duration_in_steps
        if start < 0:
            raise ValueError("Clamp duration begins before the epoch")
        return cls.between(start, stop)

    def is_active(self, time_step: int) -> bool:
        return self.window.contains(time_step)

    @property
    def duration_in_steps(self) -> int:
        return self.window.duration


def author_tms_pulse(
    labeled_dose: int,
    protocol: EpochProtocol | None = None,
) -> ConjunctionClamp | None:
    """Translate the public implementation's inclusive TMS dose label.

    The original program labels both pulse boundaries as included. A label of
    11 therefore corresponds to 12 simulated steps.
    """

    if labeled_dose < 0:
        raise ValueError("labeled_dose cannot be negative")
    if labeled_dose == 0:
        return None
    return ConjunctionClamp.ending_with_stimulus(
        duration_in_steps=labeled_dose + 1,
        protocol=protocol,
    )


@dataclass(frozen=True)
class NetworkTrajectory:
    """Time-by-unit activity recorded during one epoch."""

    feature_activity: np.ndarray
    conjunction_activity: np.ndarray


def run_epoch(
    model: PlasticAttractor,
    inputs: np.ndarray,
    protocol: EpochProtocol,
    *,
    perturbation: ConjunctionClamp | None = None,
    learn: bool | None = None,
) -> NetworkTrajectory:
    """Reset activity, run every time step, and record the trajectory."""

    input_array = np.asarray(inputs, dtype=float)
    expected_shape = (
        protocol.number_of_steps,
        model.number_of_feature_units,
    )
    if input_array.shape != expected_shape:
        raise ValueError(
            f"inputs must have shape {expected_shape}; received {input_array.shape}"
        )

    # Activity resets at each epoch. Fast and slow weights keep learning.
    model.reset_activity()

    feature_history = np.empty(
        (protocol.number_of_steps, model.number_of_feature_units)
    )
    conjunction_history = np.empty(
        (protocol.number_of_steps, model.number_of_conjunction_units)
    )

    for time_step in range(protocol.number_of_steps):
        clamp_is_active = (
            perturbation is not None and perturbation.is_active(time_step)
        )
        state = model.step(
            input_array[time_step],
            clamp_conjunctions=clamp_is_active,
            learn=learn,
        )
        feature_history[time_step] = state.feature_activity
        conjunction_history[time_step] = state.conjunction_activity

    return NetworkTrajectory(feature_history, conjunction_history)


@dataclass(frozen=True)
class BlockedExperimentConfig:
    """Every choice needed to reproduce one alternating-block simulation."""

    seed: int = 0
    number_of_blocks: int = 20
    protocol: EpochProtocol = field(default_factory=EpochProtocol)
    model_parameters: ModelParameters = field(default_factory=ModelParameters)
    perturbation: ConjunctionClamp | None = None
    learn_during_trials: bool = True

    def __post_init__(self) -> None:
        if self.number_of_blocks <= 0:
            raise ValueError("number_of_blocks must be positive")
        if self.number_of_blocks % 2:
            raise ValueError("number_of_blocks must be even")
        if (
            self.perturbation is not None
            and self.perturbation.window.stop > self.protocol.number_of_steps
        ):
            raise ValueError("perturbation window must fit inside the epoch")


@dataclass(frozen=True)
class TrialResult:
    """Behavior and neural activity recorded for one stimulus trial."""

    seed: int
    block_index: int
    trial_index_in_block: int
    task: Task
    stimulus: Stimulus
    correct_response: Feature
    chosen_response: Feature | None
    reaction_time_in_steps: float
    perturbation: ConjunctionClamp | None
    trajectory: NetworkTrajectory

    @property
    def correct(self) -> bool:
        return (
            self.chosen_response is not None
            and self.chosen_response == self.correct_response
        )

    @property
    def congruent(self) -> bool:
        return is_congruent(self.stimulus)


@dataclass(frozen=True)
class ExperimentResult:
    """Trial-level data and final summaries from one blocked experiment."""

    config: BlockedExperimentConfig
    trials: tuple[TrialResult, ...]
    amplifying_eigenvalue_count_by_block: tuple[int, ...]
    final_combined_weights: np.ndarray


def _instruction_vector(
    task: Task,
    relevant_feature: Feature,
    response: Feature,
) -> np.ndarray:
    """Represent one rule mapping, such as green to left."""

    vector = np.full(NUMBER_OF_FEATURE_UNITS, -1.0)
    vector[list(IRRELEVANT_FEATURES_BY_TASK[task])] = 0.5
    vector[relevant_feature] = 1.0
    vector[response] = 1.0
    return vector


def _instruction_vectors(task: Task) -> tuple[np.ndarray, np.ndarray]:
    """Build the two mappings needed to teach one task."""

    first_feature, second_feature = FEATURES_BY_TASK[task]
    return (
        _instruction_vector(
            task,
            first_feature,
            RESPONSE_BY_FEATURE[first_feature],
        ),
        _instruction_vector(
            task,
            second_feature,
            RESPONSE_BY_FEATURE[second_feature],
        ),
    )


def _instruction_epoch(
    instruction: np.ndarray,
    protocol: EpochProtocol,
) -> np.ndarray:
    inputs = np.full((protocol.number_of_steps, NUMBER_OF_FEATURE_UNITS), -1.0)
    window = protocol.instruction_window
    inputs[window.start : window.stop] = instruction
    return inputs


def _stimulus_vector(stimulus: Stimulus) -> np.ndarray:
    """Activate the presented color and shape."""

    vector = np.zeros(NUMBER_OF_FEATURE_UNITS)
    vector[stimulus.color] = 1.0
    vector[stimulus.shape] = 1.0
    return vector


def _trial_epoch(stimulus: Stimulus, protocol: EpochProtocol) -> np.ndarray:
    """Build the complete external-input schedule for one trial."""

    inputs = np.zeros((protocol.number_of_steps, NUMBER_OF_FEATURE_UNITS))
    inputs[: protocol.stimulus_window.start] = -1.0

    stimulus_window = protocol.stimulus_window
    inputs[stimulus_window.start : stimulus_window.stop] = _stimulus_vector(
        stimulus
    )
    inputs[protocol.response_window.stop :] = -1.0
    return inputs


def _measure_response(
    trajectory: NetworkTrajectory,
    protocol: EpochProtocol,
) -> tuple[Feature | None, float]:
    """Apply the public implementation's global-peak and 98% response rule."""

    eligible_activity = trajectory.feature_activity[
        protocol.response_search_start :, list(RESPONSE_FEATURES)
    ]
    global_peak = float(eligible_activity.max())
    if not np.isfinite(global_peak) or global_peak <= 0.0:
        return None, float("nan")

    # Transposing preserves the original response-first scan: all left-unit
    # crossings are considered before all right-unit crossings.
    above_threshold = np.argwhere(eligible_activity.T > 0.98 * global_peak)
    if not above_threshold.size:
        return None, float("nan")

    response_index, first_time = above_threshold[0]
    return RESPONSE_FEATURES[int(response_index)], float(first_time)


def _run_trial(
    model: PlasticAttractor,
    config: BlockedExperimentConfig,
    block_index: int,
    trial_index: int,
    task: Task,
    stimulus: Stimulus,
) -> TrialResult:
    trajectory = run_epoch(
        model,
        _trial_epoch(stimulus, config.protocol),
        config.protocol,
        perturbation=config.perturbation,
        learn=config.learn_during_trials,
    )
    chosen_response, reaction_time = _measure_response(
        trajectory,
        config.protocol,
    )

    return TrialResult(
        seed=config.seed,
        block_index=block_index,
        trial_index_in_block=trial_index,
        task=task,
        stimulus=stimulus,
        correct_response=correct_response(task, stimulus),
        chosen_response=chosen_response,
        reaction_time_in_steps=reaction_time,
        perturbation=config.perturbation,
        trajectory=trajectory,
    )


def run_blocked_experiment(config: BlockedExperimentConfig) -> ExperimentResult:
    """Run the alternating color-block and shape-block experiment."""

    # One generator controls initialization, order, and noise reproducibly.
    random_generator = np.random.RandomState(config.seed)
    model = PlasticAttractor(
        parameters=config.model_parameters,
        random_generator=random_generator,
    )

    trials: list[TrialResult] = []
    eigenvalue_counts: list[int] = []

    instruction_order = np.array([0, 1])
    stimulus_order = np.tile(
        np.arange(len(ALL_STIMULI)),
        config.protocol.repeats_per_stimulus,
    )

    for block_index in range(config.number_of_blocks):
        task = Task.COLOR if block_index % 2 == 0 else Task.SHAPE
        instructions = _instruction_vectors(task)

        # Every block stays balanced while the presentation order changes.
        random_generator.shuffle(instruction_order)
        random_generator.shuffle(stimulus_order)

        # Teach both mappings for the current task before presenting stimuli.
        for instruction_index in instruction_order:
            run_epoch(
                model,
                _instruction_epoch(
                    instructions[int(instruction_index)],
                    config.protocol,
                ),
                config.protocol,
            )

        # Four stimuli repeated three times produce 12 trials per block.
        for trial_index, stimulus_index in enumerate(stimulus_order):
            trials.append(
                _run_trial(
                    model=model,
                    config=config,
                    block_index=block_index,
                    trial_index=trial_index,
                    task=task,
                    stimulus=ALL_STIMULI[int(stimulus_index)],
                )
            )

        eigenvalue_counts.append(model.number_of_amplifying_eigenvalues())

    return ExperimentResult(
        config=config,
        trials=tuple(trials),
        amplifying_eigenvalue_count_by_block=tuple(eigenvalue_counts),
        final_combined_weights=model.combined_weights,
    )


def run_baseline(
    seed: int = 0,
    number_of_blocks: int = 20,
) -> ExperimentResult:
    """Run the published baseline using only two ordinary arguments.

    This is the easiest entry point for new Python users. The configuration
    class remains available when an experiment later needs more settings.
    """

    config = BlockedExperimentConfig(
        seed=seed,
        number_of_blocks=number_of_blocks,
    )
    return run_blocked_experiment(config)
