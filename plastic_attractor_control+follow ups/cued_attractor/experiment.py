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

from .model import ModelParameters, PlasticAttractor
from .task import (
    ALL_STIMULI,
    Stimulus,
    Task,
    Vocabulary,
    build_vocabulary,
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
    cue: int
    stimulus: Stimulus


def _balanced_shuffled_deck(
    task: Task,
    permutation_repeats: int,
    vocabulary: Vocabulary,
    random_generator: np.random.RandomState,
    *,
    allowed_cues: tuple[int, ...] | None = None,
    omit_combo: tuple[int, Stimulus] | None = None,
) -> list[tuple[int, Stimulus]]:
    """Every (cue x stimulus) combo for task, tiled permutation_repeats times + shuffled.

    allowed_cues restricts the cue pool below cues_for_task(vocabulary, task)
    (Control: novel-cue generalization -- taught on a subset, tested on all).
    omit_combo drops one specific (cue, stimulus) pairing from the deck
    (Control: omitted-combination generalization).
    """

    cues = allowed_cues if allowed_cues is not None else cues_for_task(vocabulary, task)
    combos = [
        (cue, stimulus)
        for cue in cues
        for stimulus in ALL_STIMULI
        if omit_combo is None or (cue, stimulus) != omit_combo
    ]
    if not combos:
        raise ValueError(
            'No (cue, stimulus) combinations remain after allowed_cues/omit_combo'
        )
    combos = combos * permutation_repeats
    order = random_generator.permutation(len(combos))
    return [combos[i] for i in order]


def practice_block_plan(
    task: Task,
    permutation_repeats: int,
    vocabulary: Vocabulary,
    random_generator: np.random.RandomState,
    *,
    allowed_cues: tuple[int, ...] | None = None,
    omit_combo: tuple[int, Stimulus] | None = None,
) -> list[PlannedTrial]:
    """Plan one practice block: every (cue x stimulus) combo, tiled + shuffled.

    Runs through the full permutation permutation_repeats times, so every cue
    is taught on every stimulus an equal number of times. allowed_cues and
    omit_combo implement Controls 2 and 3; see _balanced_shuffled_deck.
    """

    if permutation_repeats <= 0:
        raise ValueError('practice_permutation_repeats must be positive')

    deck = _balanced_shuffled_deck(
        task, permutation_repeats, vocabulary, random_generator,
        allowed_cues=allowed_cues, omit_combo=omit_combo,
    )
    return [PlannedTrial(task, cue, stimulus) for cue, stimulus in deck]


def switching_practice_block_plan(
    switch_probability: float,
    permutation_repeats: int,
    vocabulary: Vocabulary,
    random_generator: np.random.RandomState,
    *,
    allowed_cues_by_task: dict[Task, tuple[int, ...]] | None = None,
    omit_combo: tuple[int, Stimulus] | None = None,
) -> list[PlannedTrial]:
    """Plan one practice block that may switch rule mid-block, like a real block.

    Unlike real_block_plan's i.i.d. cue draws, this keeps practice's balance
    guarantee: one shuffled (cue x stimulus) deck per rule, each covering the
    full permutation permutation_repeats times, so every combo is still taught
    exactly permutation_repeats times per rule even while switching.
    allowed_cues_by_task and omit_combo implement Controls 2 and 3.
    """

    if permutation_repeats <= 0:
        raise ValueError('practice_permutation_repeats must be positive')
    if not 0.0 <= switch_probability <= 1.0:
        raise ValueError('switch_probability must lie between 0 and 1')

    decks = {
        task: _balanced_shuffled_deck(
            task, permutation_repeats, vocabulary, random_generator,
            allowed_cues=(
                allowed_cues_by_task.get(task) if allowed_cues_by_task is not None else None
            ),
            omit_combo=omit_combo,
        )
        for task in PRACTICE_TASKS
    }
    total_trials = sum(len(deck) for deck in decks.values())

    current_task = PRACTICE_TASKS[random_generator.randint(len(PRACTICE_TASKS))]
    plan: list[PlannedTrial] = []
    for _ in range(total_trials):
        if random_generator.rand() < switch_probability:
            other = PRACTICE_TASKS[1 - PRACTICE_TASKS.index(current_task)]
            if decks[other]:
                current_task = other
        if not decks[current_task]:
            current_task = next(task for task in PRACTICE_TASKS if decks[task])

        cue, stimulus = decks[current_task].pop(0)
        plan.append(PlannedTrial(current_task, cue, stimulus))

    return plan


def _draw_cue_avoiding_combo(
    task_cues: tuple[int, ...],
    stimulus: Stimulus,
    omit_combo: tuple[int, Stimulus] | None,
    random_generator: np.random.RandomState,
) -> int:
    """Draw one cue uniformly from task_cues, resampling to avoid recreating
    omit_combo when it's paired with this trial's stimulus."""

    if omit_combo is None or omit_combo[1] != stimulus:
        return task_cues[random_generator.randint(len(task_cues))]

    remaining = tuple(cue for cue in task_cues if cue != omit_combo[0])
    if not remaining:
        raise ValueError(
            'omit_practice_combo leaves no valid cue for this stimulus in a '
            'block drawn from real_block_plan (e.g. performance practice); '
            'set include_performance_practice=False or allow another cue '
            'for this rule.'
        )
    return remaining[random_generator.randint(len(remaining))]


def real_block_plan(
    switch_probability: float,
    num_trials: int,
    vocabulary: Vocabulary,
    random_generator: np.random.RandomState,
    initial_task: Task | None = None,
    *,
    allowed_cues_by_task: dict[Task, tuple[int, ...]] | None = None,
    omit_combo: tuple[int, Stimulus] | None = None,
    shuffle_cues: bool = False,
) -> list[PlannedTrial]:
    """Plan one real block: rule switches w.p. switch_probability, cue signals it.

    num_trials must be a multiple of 4 (four stimuli tiled evenly). The first
    rule is initial_task if given, otherwise random; each later trial switches
    rule with the given probability. The cue is drawn uniformly from the
    active rule's cues (allowed_cues_by_task narrows that pool; omit_combo
    resamples the cue rather than let it recreate the omitted pairing --
    Controls 2 and 3, also usable on performance-practice blocks so a
    "novel" cue or combo stays genuinely unseen until the real switching
    blocks below in run_switching_experiment). Stimulus identity is drawn
    first and is independent of the rule.

    shuffle_cues reassigns the drawn cues across trials after the fact, so
    the cue on a given trial no longer reliably signals which rule is active
    (Control 1: is the network reading the cue, or just the stimulus?).
    Leave allowed_cues_by_task and omit_combo unset when shuffle_cues=True --
    real (test) blocks are unrestricted in Controls 2/3, only practice is.
    """

    if num_trials <= 0 or num_trials % 4:
        raise ValueError('num_trials must be a positive multiple of 4')
    if not 0.0 <= switch_probability <= 1.0:
        raise ValueError('switch_probability must lie between 0 and 1')

    first_task = (
        initial_task
        if initial_task is not None
        else PRACTICE_TASKS[random_generator.randint(2)]
    )
    tasks: list[Task] = [first_task]
    for _ in range(1, num_trials):
        if random_generator.rand() < switch_probability:
            other = 1 - PRACTICE_TASKS.index(tasks[-1])
            tasks.append(PRACTICE_TASKS[other])
        else:
            tasks.append(tasks[-1])

    stimulus_indices = np.tile(np.arange(len(ALL_STIMULI)), num_trials // 4)
    random_generator.shuffle(stimulus_indices)
    stimuli = [ALL_STIMULI[int(index)] for index in stimulus_indices]

    cues = []
    for task, stimulus in zip(tasks, stimuli):
        task_cues = (
            allowed_cues_by_task.get(task, cues_for_task(vocabulary, task))
            if allowed_cues_by_task is not None
            else cues_for_task(vocabulary, task)
        )
        cues.append(_draw_cue_avoiding_combo(task_cues, stimulus, omit_combo, random_generator))

    if shuffle_cues:
        cues = list(cues)
        random_generator.shuffle(cues)

    return [
        PlannedTrial(task, cue, stimulus)
        for task, cue, stimulus in zip(tasks, cues, stimuli)
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
    # matches the original implementation: one instruction trial per (cue x
    # stimulus) combination, no repeats.
    practice_permutation_repeats: int = 1
    practice_switch_probability: float = 0.0
    # after instruction, one performance-practice block of num_trials real
    # trials per rule: actual behaviour, no teaching drive, learning stays on.
    include_performance_practice: bool = True
    num_trials: int = 48
    switch_probs: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75) * 2
    protocol: EpochProtocol = field(default_factory=EpochProtocol)
    model_parameters: ModelParameters = field(default_factory=ModelParameters)
    perturbation: ConjunctionClamp | None = None
    learn_during_trials: bool = True

    # --- Experimental controls, all off (baseline) by default --- #
    # Control 1: shuffle each real block's cue assignment after the fact, so
    # the cue no longer reliably signals the active rule at test.
    shuffle_cues_test: bool = False
    # Control 2: restrict the taught cue pool per rule during practice (both
    # the instruction blocks and, if enabled, performance practice), e.g.
    # {Task.COLOR: (color_cue_0,), Task.SHAPE: (shape_cue_0,)}, so the other
    # cue(s) are novel at the real switching blocks. None = no restriction.
    practice_cue_restriction: dict[Task, tuple[int, ...]] | None = None
    # Control 3: omit one (cue, stimulus) pairing from practice (instruction
    # and, if enabled, performance practice), so the real switching blocks
    # test a combination the network never saw during training.
    omit_practice_combo: tuple[int, Stimulus] | None = None

    def __post_init__(self) -> None:
        if self.practice_switch_probability == 0.0:
            if not 1 <= self.num_practice_blocks <= len(PRACTICE_TASKS):
                raise ValueError(
                    f'num_practice_blocks must be between 1 and {len(PRACTICE_TASKS)} '
                    'when practice_switch_probability is 0'
                )
        elif self.num_practice_blocks < 1:
            raise ValueError('num_practice_blocks must be positive')
        if self.practice_permutation_repeats <= 0:
            raise ValueError('practice_permutation_repeats must be positive')
        if not 0.0 <= self.practice_switch_probability <= 1.0:
            raise ValueError('practice_switch_probability must lie between 0 and 1')
        if not self.switch_probs:
            raise ValueError('switch_probs needs at least one real block')
        if (
            self.perturbation is not None
            and self.perturbation.window.stop > self.protocol.number_of_steps
        ):
            raise ValueError('perturbation window must fit inside the epoch')
        if self.practice_cue_restriction is not None:
            vocabulary = build_vocabulary(self.model_parameters.num_cues_per_rule)
            for task, cues in self.practice_cue_restriction.items():
                if task not in PRACTICE_TASKS:
                    raise ValueError(f'practice_cue_restriction has unknown task {task!r}')
                if not cues:
                    raise ValueError(f'practice_cue_restriction for {task!r} must not be empty')
                valid_cues = set(cues_for_task(vocabulary, task))
                if not set(cues) <= valid_cues:
                    raise ValueError(
                        f'practice_cue_restriction for {task!r} must be a subset of {valid_cues}'
                    )
        if self.omit_practice_combo is not None:
            cue, stimulus = self.omit_practice_combo
            if not isinstance(stimulus, Stimulus):
                raise ValueError('omit_practice_combo must be (cue_index, Stimulus)')

    @property
    def practice_trials(self) -> int:
        """Trials in one practice block (double under mid-block switching, since
        both rules are then taught within a single block)."""

        per_rule = (
            self.practice_permutation_repeats
            * self.model_parameters.num_cues_per_rule
            * len(ALL_STIMULI)
        )
        return per_rule * (2 if self.practice_switch_probability > 0.0 else 1)

    @property
    def number_of_blocks(self) -> int:
        performance_practice_blocks = (
            len(PRACTICE_TASKS) if self.include_performance_practice else 0
        )
        return (
            self.num_practice_blocks
            + performance_practice_blocks
            + len(self.switch_probs)
        )


@dataclass(frozen=True)
class TrialResult:
    """Behaviour and neural activity recorded for one trial."""

    seed: int
    block_index: int
    trial_index_in_block: int
    is_practice: bool
    is_instruction: bool  # True only on trials with the artificial teaching drive
    switch_probability: float | None  # None on practice trials
    task: Task
    cue: int
    stimulus: Stimulus
    correct_response: int
    chosen_response: int | None
    reaction_time_in_steps: float
    transition_type: TransitionType
    perturbation: ConjunctionClamp | None
    trajectory: NetworkTrajectory
    combined_weights: np.ndarray  # snapshot of W after this trial, shape (num_features, num_conjunction_units)

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


def _stimulus_vector(stimulus: Stimulus, vocabulary: Vocabulary) -> np.ndarray:
    """Activate the presented colour and shape units."""

    vector = np.zeros(vocabulary.number_of_features)
    vector[stimulus.color] = 1.0
    vector[stimulus.shape] = 1.0
    return vector


def _cue_vector(cue: int, vocabulary: Vocabulary) -> np.ndarray:
    """Activate the single cue unit for this trial."""

    vector = np.zeros(vocabulary.number_of_features)
    vector[cue] = 1.0
    return vector


def _teaching_drive(correct: int, vocabulary: Vocabulary) -> np.ndarray:
    """Drive on the action units teaching the correct response.

    +1 on the correct action unit, -1 on the others (old action_teach rows
    [[1, -1], [-1, 1]]). Returned in vocabulary.response_features order.
    """

    return np.array(
        [1.0 if action == correct else -1.0 for action in vocabulary.response_features]
    )


def _trial_epoch(
    planned: PlannedTrial,
    apply_teaching: bool,
    protocol: EpochProtocol,
    vocabulary: Vocabulary,
) -> np.ndarray:
    """Build the full external-input schedule (steps x features) for one trial."""

    inputs = np.zeros((protocol.number_of_steps, vocabulary.number_of_features))

    # isi: negative drive to every feature before the stimulus and after the
    # response window (the response window itself stays at zero input).
    inputs[: protocol.stimulus_window.start] = -1.0
    inputs[protocol.response_window.stop :] = -1.0

    # stimulus and its rule cue, presented together.
    stimulus_window = protocol.stimulus_window
    inputs[stimulus_window.start : stimulus_window.stop] = (
        _stimulus_vector(planned.stimulus, vocabulary)
        + _cue_vector(planned.cue, vocabulary)
    )

    # instruction trials teach the response across the settling window,
    # overriding whatever base input sits on the action units there. held past
    # the stimulus because the Hebbian update runs every step and would
    # otherwise reinforce whatever attractor the network drifted into while
    # settling. performance-practice and real trials leave this off, so the
    # response is the network's own, unforced choice.
    #
    # deliberately does NOT also drive the task-irrelevant colour/shape pair
    # to a neutral value here: doing so held it at a constant, trial-generic
    # 0.5 for the entire ~300-step teaching window, versus the relevant
    # feature's brief ~50-step flash during the stimulus window -- that 6x
    # exposure imbalance was swamping the real signal in the Hebbian update
    # and dragging real-block accuracy down to chance (verified by ablation:
    # baseline real-block accuracy 0.50 with the override vs 0.61 without).
    if apply_teaching:
        correct = correct_response(vocabulary, planned.task, planned.stimulus)
        teaching_window = protocol.teaching_window
        inputs[
            teaching_window.start : teaching_window.stop,
            list(vocabulary.response_features),
        ] = _teaching_drive(correct, vocabulary)

    return inputs


def _measure_response(
    trajectory: NetworkTrajectory,
    protocol: EpochProtocol,
    vocabulary: Vocabulary,
) -> tuple[int | None, float]:
    """Apply the flat script's global-peak and 98%-of-peak response rule."""

    eligible_activity = trajectory.feature_activity[
        protocol.response_search_start :, list(vocabulary.response_features)
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
    return vocabulary.response_features[int(response_index)], float(first_time)


def _run_trial(
    model: PlasticAttractor,
    config: SwitchingExperimentConfig,
    vocabulary: Vocabulary,
    block_index: int,
    trial_index: int,
    is_practice: bool,
    apply_teaching: bool,
    switch_probability: float | None,
    planned: PlannedTrial,
    transition_type: TransitionType,
) -> TrialResult:
    trajectory = run_epoch(
        model,
        _trial_epoch(planned, apply_teaching, config.protocol, vocabulary),
        config.protocol,
        perturbation=config.perturbation,
        learn=config.learn_during_trials,
    )
    chosen_response, reaction_time = _measure_response(
        trajectory,
        config.protocol,
        vocabulary,
    )

    return TrialResult(
        seed=config.seed,
        block_index=block_index,
        trial_index_in_block=trial_index,
        is_practice=is_practice,
        is_instruction=apply_teaching,
        switch_probability=switch_probability,
        task=planned.task,
        cue=planned.cue,
        stimulus=planned.stimulus,
        correct_response=correct_response(vocabulary, planned.task, planned.stimulus),
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
    vocabulary: Vocabulary,
    block_index: int,
    is_practice: bool,
    apply_teaching: bool,
    switch_probability: float | None,
    plan: list[PlannedTrial],
) -> list[TrialResult]:
    return [
        _run_trial(
            model=model,
            config=config,
            vocabulary=vocabulary,
            block_index=block_index,
            trial_index=trial_index,
            is_practice=is_practice,
            apply_teaching=apply_teaching,
            switch_probability=switch_probability,
            planned=planned,
            transition_type=_transition_type(plan, trial_index),
        )
        for trial_index, planned in enumerate(plan)
    ]


def run_switching_experiment(
    config: SwitchingExperimentConfig,
) -> ExperimentResult:
    """Run instruction, then performance practice, then the cued real blocks."""

    vocabulary = build_vocabulary(config.model_parameters.num_cues_per_rule)

    # One generator controls initialization, trial order, and noise reproducibly.
    random_generator = np.random.RandomState(config.seed)
    model = PlasticAttractor(
        parameters=config.model_parameters,
        random_generator=random_generator,
    )

    trials: list[TrialResult] = []
    eigenvalue_counts: list[int] = []

    # instruction blocks: teach with the action-teaching drive. non-switching
    # blocks each teach one fixed rule; switching blocks mix both rules.
    # practice_cue_restriction / omit_practice_combo (Controls 2 and 3) apply
    # here so the restricted cue or omitted combo is never taught.
    for block_index in range(config.num_practice_blocks):
        if config.practice_switch_probability > 0.0:
            plan = switching_practice_block_plan(
                config.practice_switch_probability,
                config.practice_permutation_repeats,
                vocabulary, random_generator,
                allowed_cues_by_task=config.practice_cue_restriction,
                omit_combo=config.omit_practice_combo,
            )
        else:
            task = PRACTICE_TASKS[block_index]
            allowed_cues = (
                config.practice_cue_restriction.get(task)
                if config.practice_cue_restriction is not None else None
            )
            plan = practice_block_plan(
                task, config.practice_permutation_repeats,
                vocabulary, random_generator,
                allowed_cues=allowed_cues,
                omit_combo=config.omit_practice_combo,
            )
        trials.extend(
            _run_block(
                model, config, vocabulary, block_index,
                is_practice=True, apply_teaching=True,
                switch_probability=None, plan=plan,
            )
        )
        eigenvalue_counts.append(model.number_of_amplifying_eigenvalues())

    # performance-practice blocks: one fixed-rule block per rule, num_trials
    # trials each. no teaching drive, so the response is the network's own;
    # learning stays on (learn_during_trials), so this is still training --
    # the same restriction/omission carries over here too, otherwise a
    # "novel" cue or combo would already have been trained on by the time
    # the real switching blocks run, defeating Controls 2 and 3.
    if config.include_performance_practice:
        for practice_index, task in enumerate(PRACTICE_TASKS):
            block_index = config.num_practice_blocks + practice_index
            plan = real_block_plan(
                switch_probability=0.0,
                num_trials=config.num_trials,
                vocabulary=vocabulary,
                random_generator=random_generator,
                initial_task=task,
                allowed_cues_by_task=config.practice_cue_restriction,
                omit_combo=config.omit_practice_combo,
            )
            trials.extend(
                _run_block(
                    model, config, vocabulary, block_index,
                    is_practice=True, apply_teaching=False,
                    switch_probability=None, plan=plan,
                )
            )
            eigenvalue_counts.append(model.number_of_amplifying_eigenvalues())

    # real (test) blocks: no teaching, the cue signals the rule that varies
    # trial by trial. learning stays on so the weights keep evolving. always
    # unrestricted and complete (every cue, every combo) -- Controls 2 and 3
    # only touch what's taught, not what's tested. shuffle_cues_test (Control
    # 1) applies only here, breaking the cue-rule correspondence at test.
    performance_practice_blocks = (
        len(PRACTICE_TASKS) if config.include_performance_practice else 0
    )
    for real_index, switch_probability in enumerate(config.switch_probs):
        block_index = (
            config.num_practice_blocks + performance_practice_blocks + real_index
        )
        plan = real_block_plan(
            switch_probability, config.num_trials, vocabulary, random_generator,
            shuffle_cues=config.shuffle_cues_test,
        )
        trials.extend(
            _run_block(
                model, config, vocabulary, block_index,
                is_practice=False, apply_teaching=False,
                switch_probability=switch_probability,
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
    practice_permutation_repeats: int = 1,
    switch_probs: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75) * 2,
) -> ExperimentResult:
    """Run the standard cued-switching design with plain arguments.

    The easiest entry point; the configuration class stays available when an
    experiment needs more settings (TMS, altered timing, new parameters).
    """

    config = SwitchingExperimentConfig(
        seed=seed,
        num_trials=num_trials,
        practice_permutation_repeats=practice_permutation_repeats,
        switch_probs=switch_probs,
    )
    return run_switching_experiment(config)
