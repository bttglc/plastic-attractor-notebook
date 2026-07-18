"""Timing, inputs, results, and the cued task-switching experiment.

This module translates experimental concepts into external inputs for the
network. It is the natural place to change the task design later.

It reshapes published_model/plastic_attractor/experiment.py for the cued
paradigm: instead of re-teaching the rule with a separate instruction epoch
every block, the network is taught once in two practice blocks (an action-
teaching drive on the response units), then runs real blocks where the rule
varies trial by trial and is signalled only by a cue on the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .model import NUMBER_OF_FEATURE_UNITS, ModelParameters, PlasticAttractor
from .task import (
    ALL_STIMULI,
    Cue,
    Feature,
    RESPONSE_FEATURES,
    Stimulus,
    Task,
    correct_response,
    cues_for_task,
    is_congruent,
)


# The two practice blocks teach one rule each, in this order.
PRACTICE_TASKS = (Task.COLOR, Task.SHAPE)


class TransitionType(IntEnum):
    """How a real trial relates to the previous one in its block."""

    FIRST_TRIAL = -1          # first trial of a block, no predecessor
    CUE_REPEAT = 0            # same cue as before (so same rule)
    CUE_SWITCH_RULE_REPEAT = 1  # different cue, same rule
    RULE_SWITCH = 2           # different rule


# The experiment dataclasses keep related settings and results together.
@dataclass(frozen=True)
class TimeWindow:
    """A half-open interval: start is included and stop is excluded."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError('A time window needs 0 <= start < stop')

    def contains(self, time_step: int) -> bool:
        return self.start <= time_step < self.stop

    @property
    def duration(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class EpochProtocol:
    """Per-trial timing, matching the flat attractor_rnn.py schedule.

    Each trial runs number_of_steps steps: inter-stimulus interval (a negative
    drive pushing activity down) before the stimulus window and after the
    response window, the stimulus-and-cue presented together in the stimulus
    window, and the response window left at zero input so the network settles on
    a choice. On practice trials the teaching_window drives the response units.
    """

    number_of_steps: int = 400
    stimulus_window: TimeWindow = field(
        default_factory=lambda: TimeWindow(51, 101)
    )
    response_window: TimeWindow = field(
        default_factory=lambda: TimeWindow(101, 351)
    )
    teaching_window: TimeWindow = field(
        default_factory=lambda: TimeWindow(50, 351)
    )
    response_search_start: int = 110

    def __post_init__(self) -> None:
        if self.number_of_steps <= 0:
            raise ValueError('number_of_steps must be positive')
        for window in (
            self.stimulus_window,
            self.response_window,
            self.teaching_window,
        ):
            if window.stop > self.number_of_steps:
                raise ValueError('Protocol windows must fit inside the epoch')
        if not 0 <= self.response_search_start < self.number_of_steps:
            raise ValueError('response_search_start must lie inside the epoch')


@dataclass(frozen=True)
class ConjunctionClamp:
    """Force every conjunction unit to maximum activity in a time window."""

    window: TimeWindow

    @classmethod
    def between(cls, start: int, stop: int) -> 'ConjunctionClamp':
        return cls(TimeWindow(start, stop))

    @classmethod
    def ending_with_stimulus(
        cls,
        duration_in_steps: int,
        protocol: 'EpochProtocol | None' = None,
    ) -> 'ConjunctionClamp':
        if duration_in_steps <= 0:
            raise ValueError('duration_in_steps must be positive')

        timing = protocol or EpochProtocol()
        stop = timing.stimulus_window.stop
        start = stop - duration_in_steps
        if start < 0:
            raise ValueError('Clamp duration begins before the epoch')
        return cls.between(start, stop)

    def is_active(self, time_step: int) -> bool:
        return self.window.contains(time_step)

    @property
    def duration_in_steps(self) -> int:
        return self.window.duration


def author_tms_pulse(
    labeled_dose: int,
    protocol: 'EpochProtocol | None' = None,
) -> 'ConjunctionClamp | None':
    """Translate the flat script's inclusive TMS dose label into a clamp.

    The original program clamps the conjunction units over t in [TMS_start, 100]
    with both boundaries included, so a labelled dose (100 - TMS_start)
    corresponds to that many + 1 simulated steps. A dose of 0 means no TMS.
    """

    if labeled_dose < 0:
        raise ValueError('labeled_dose cannot be negative')
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
            f'inputs must have shape {expected_shape}; received {input_array.shape}'
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


# One planned trial: which rule is active, which cue signals it, and the
# stimulus shown. The plan is drawn up before the network runs so transition
# types can be read straight off consecutive entries.
@dataclass(frozen=True)
class PlannedTrial:
    task: Task
    cue: Cue
    stimulus: Stimulus


def practice_block_plan(
    task: Task,
    practice_trials: int,
    random_generator: np.random.RandomState,
) -> list[PlannedTrial]:
    """Plan one practice block: every (2 cues x 4 stimuli) combo, tiled + shuffled.

    practice_trials must be a multiple of 8 so both cues are taught on every
    stimulus an equal number of times.
    """

    if practice_trials <= 0 or practice_trials % 8:
        raise ValueError('practice_trials must be a positive multiple of 8')

    cues = cues_for_task(task)
    combos = [
        (cue, stimulus)
        for cue in cues
        for stimulus in ALL_STIMULI
    ]
    combos = combos * (practice_trials // 8)

    order = random_generator.permutation(len(combos))
    return [PlannedTrial(task, combos[i][0], combos[i][1]) for i in order]


def real_block_plan(
    switch_probability: float,
    num_trials: int,
    random_generator: np.random.RandomState,
) -> list[PlannedTrial]:
    """Plan one real block: rule switches w.p. switch_probability, cue signals it.

    num_trials must be a multiple of 4 (four stimuli tiled evenly). The first
    rule is random; each later trial switches rule with the given probability.
    The cue is drawn uniformly from the active rule's two cues, so a rule repeat
    can still be a cue switch. Stimulus identity is independent of the rule.
    """

    if num_trials <= 0 or num_trials % 4:
        raise ValueError('num_trials must be a positive multiple of 4')
    if not 0.0 <= switch_probability <= 1.0:
        raise ValueError('switch_probability must lie between 0 and 1')

    tasks: list[Task] = [PRACTICE_TASKS[random_generator.randint(2)]]
    for _ in range(1, num_trials):
        if random_generator.rand() < switch_probability:
            other = 1 - PRACTICE_TASKS.index(tasks[-1])
            tasks.append(PRACTICE_TASKS[other])
        else:
            tasks.append(tasks[-1])

    cues = [
        cues_for_task(task)[random_generator.randint(2)] for task in tasks
    ]

    stimulus_indices = np.tile(np.arange(len(ALL_STIMULI)), num_trials // 4)
    random_generator.shuffle(stimulus_indices)

    return [
        PlannedTrial(task, cue, ALL_STIMULI[int(index)])
        for task, cue, index in zip(tasks, cues, stimulus_indices)
    ]


def _transition_type(plan: list[PlannedTrial], index: int) -> TransitionType:
    """Classify trial `index` relative to its predecessor in the same block."""

    if index == 0:
        return TransitionType.FIRST_TRIAL
    previous, current = plan[index - 1], plan[index]
    if current.task != previous.task:
        return TransitionType.RULE_SWITCH
    if current.cue != previous.cue:
        return TransitionType.CUE_SWITCH_RULE_REPEAT
    return TransitionType.CUE_REPEAT


@dataclass(frozen=True)
class SwitchingExperimentConfig:
    """Every choice needed to reproduce one cued-switching simulation."""

    seed: int = 0
    num_practice_blocks: int = 2
    practice_trials: int = 48
    num_trials: int = 48
    switch_probs: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75) * 2
    protocol: EpochProtocol = field(default_factory=EpochProtocol)
    model_parameters: ModelParameters = field(default_factory=ModelParameters)
    perturbation: ConjunctionClamp | None = None
    learn_during_trials: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.num_practice_blocks <= len(PRACTICE_TASKS):
            raise ValueError(
                f'num_practice_blocks must be between 1 and {len(PRACTICE_TASKS)}'
            )
        if not self.switch_probs:
            raise ValueError('switch_probs needs at least one real block')
        if (
            self.perturbation is not None
            and self.perturbation.window.stop > self.protocol.number_of_steps
        ):
            raise ValueError('perturbation window must fit inside the epoch')

    @property
    def number_of_blocks(self) -> int:
        return self.num_practice_blocks + len(self.switch_probs)


@dataclass(frozen=True)
class TrialResult:
    """Behaviour and neural activity recorded for one trial."""

    seed: int
    block_index: int
    trial_index_in_block: int
    is_practice: bool
    switch_probability: float | None  # None on practice trials
    task: Task
    cue: Cue
    stimulus: Stimulus
    correct_response: Feature
    chosen_response: Feature | None
    reaction_time_in_steps: float
    transition_type: TransitionType
    perturbation: ConjunctionClamp | None
    trajectory: NetworkTrajectory
    combined_weights: np.ndarray  # snapshot of W after this trial, shape (10, 4)

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
    """Trial-level data and final summaries from one cued-switching experiment."""

    config: SwitchingExperimentConfig
    trials: tuple[TrialResult, ...]
    amplifying_eigenvalue_count_by_block: tuple[int, ...]
    final_combined_weights: np.ndarray


def _stimulus_vector(stimulus: Stimulus) -> np.ndarray:
    """Activate the presented colour and shape units."""

    vector = np.zeros(NUMBER_OF_FEATURE_UNITS)
    vector[stimulus.color] = 1.0
    vector[stimulus.shape] = 1.0
    return vector


def _cue_vector(cue: Cue) -> np.ndarray:
    """Activate the single cue unit for this trial."""

    vector = np.zeros(NUMBER_OF_FEATURE_UNITS)
    vector[int(cue)] = 1.0
    return vector


def _teaching_drive(correct: Feature) -> np.ndarray:
    """Drive on the two action units teaching the correct response.

    +1 on the correct action unit, -1 on the other (old action_teach rows
    [[1, -1], [-1, 1]]). Returned in RESPONSE_FEATURES order (action1, action2).
    """

    return np.array(
        [1.0 if action == correct else -1.0 for action in RESPONSE_FEATURES]
    )


def _trial_epoch(
    planned: PlannedTrial,
    is_practice: bool,
    protocol: EpochProtocol,
) -> np.ndarray:
    """Build the full external-input schedule (steps x features) for one trial."""

    inputs = np.zeros((protocol.number_of_steps, NUMBER_OF_FEATURE_UNITS))

    # isi: negative drive to every feature before the stimulus and after the
    # response window (the response window itself stays at zero input).
    inputs[: protocol.stimulus_window.start] = -1.0
    inputs[protocol.response_window.stop :] = -1.0

    # stimulus and its rule cue, presented together.
    stimulus_window = protocol.stimulus_window
    inputs[stimulus_window.start : stimulus_window.stop] = (
        _stimulus_vector(planned.stimulus) + _cue_vector(planned.cue)
    )

    # practice trials teach the response across the settling window, overriding
    # whatever base input sits on the action units there. held past the stimulus
    # because the Hebbian update runs every step and would otherwise reinforce
    # whatever attractor the network drifted into while settling.
    if is_practice:
        correct = correct_response(planned.task, planned.stimulus)
        teaching_window = protocol.teaching_window
        inputs[
            teaching_window.start : teaching_window.stop,
            list(RESPONSE_FEATURES),
        ] = _teaching_drive(correct)

    return inputs


def _measure_response(
    trajectory: NetworkTrajectory,
    protocol: EpochProtocol,
) -> tuple[Feature | None, float]:
    """Apply the flat script's global-peak and 98%-of-peak response rule."""

    eligible_activity = trajectory.feature_activity[
        protocol.response_search_start :, list(RESPONSE_FEATURES)
    ]
    global_peak = float(eligible_activity.max())
    if not np.isfinite(global_peak) or global_peak <= 0.0:
        return None, float('nan')

    # Transposing preserves the original response-first scan: all action-1
    # crossings are considered before all action-2 crossings.
    above_threshold = np.argwhere(eligible_activity.T > 0.98 * global_peak)
    if not above_threshold.size:
        return None, float('nan')

    response_index, first_time = above_threshold[0]
    return RESPONSE_FEATURES[int(response_index)], float(first_time)


def _run_trial(
    model: PlasticAttractor,
    config: SwitchingExperimentConfig,
    block_index: int,
    trial_index: int,
    is_practice: bool,
    switch_probability: float | None,
    planned: PlannedTrial,
    transition_type: TransitionType,
) -> TrialResult:
    trajectory = run_epoch(
        model,
        _trial_epoch(planned, is_practice, config.protocol),
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
        is_practice=is_practice,
        switch_probability=switch_probability,
        task=planned.task,
        cue=planned.cue,
        stimulus=planned.stimulus,
        correct_response=correct_response(planned.task, planned.stimulus),
        chosen_response=chosen_response,
        reaction_time_in_steps=reaction_time,
        transition_type=transition_type,
        perturbation=config.perturbation,
        trajectory=trajectory,
        combined_weights=model.combined_weights,
    )


def _run_block(
    model: PlasticAttractor,
    config: SwitchingExperimentConfig,
    block_index: int,
    is_practice: bool,
    switch_probability: float | None,
    plan: list[PlannedTrial],
) -> list[TrialResult]:
    return [
        _run_trial(
            model=model,
            config=config,
            block_index=block_index,
            trial_index=trial_index,
            is_practice=is_practice,
            switch_probability=switch_probability,
            planned=planned,
            transition_type=_transition_type(plan, trial_index),
        )
        for trial_index, planned in enumerate(plan)
    ]


def run_switching_experiment(
    config: SwitchingExperimentConfig,
) -> ExperimentResult:
    """Run the practice blocks then the cued real blocks."""

    # One generator controls initialization, trial order, and noise reproducibly.
    random_generator = np.random.RandomState(config.seed)
    model = PlasticAttractor(
        parameters=config.model_parameters,
        random_generator=random_generator,
    )

    trials: list[TrialResult] = []
    eigenvalue_counts: list[int] = []

    # practice blocks: teach one rule each with the action-teaching drive.
    for block_index in range(config.num_practice_blocks):
        task = PRACTICE_TASKS[block_index]
        plan = practice_block_plan(
            task, config.practice_trials, random_generator
        )
        trials.extend(
            _run_block(
                model, config, block_index,
                is_practice=True, switch_probability=None, plan=plan,
            )
        )
        eigenvalue_counts.append(model.number_of_amplifying_eigenvalues())

    # real blocks: no teaching, the cue signals the rule that varies trial by
    # trial. learning stays on so the weights keep evolving.
    for real_index, switch_probability in enumerate(config.switch_probs):
        block_index = config.num_practice_blocks + real_index
        plan = real_block_plan(
            switch_probability, config.num_trials, random_generator
        )
        trials.extend(
            _run_block(
                model, config, block_index,
                is_practice=False, switch_probability=switch_probability,
                plan=plan,
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
    num_trials: int = 48,
    practice_trials: int = 48,
    switch_probs: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75) * 2,
) -> ExperimentResult:
    """Run the standard cued-switching design with plain arguments.

    The easiest entry point; the configuration class stays available when an
    experiment needs more settings (TMS, altered timing, new parameters).
    """

    config = SwitchingExperimentConfig(
        seed=seed,
        num_trials=num_trials,
        practice_trials=practice_trials,
        switch_probs=switch_probs,
    )
    return run_switching_experiment(config)
