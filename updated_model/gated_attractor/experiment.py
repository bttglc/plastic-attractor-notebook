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
    IRRELEVANT_FEATURES_BY_TASK,
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
    # kept brief (50 steps), not lengthened to keep the cue "on" longer: a
    # longer window was tried so the gate would have more genuine cue-driven
    # signal to persist on, but a directly-injected stimulus/cue is also a
    # hard external floor on feature activity for as long as it's on, which
    # left gating_to_feature_gain nothing to visibly suppress until the
    # window ended (see cued-attractor-gating-units memory). Persistence
    # through the rest of the trial is now the *gate's own* job -- see
    # gating_self_weight -- so the external cue only needs to be on long
    # enough to set the correct gate winning, which this window already does.
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
    """Time-by-unit activity recorded during one epoch.

    gating_activity has shape (steps, num_gating_units); its second axis is
    length 0 when gating is disabled.
    """

    feature_activity: np.ndarray
    conjunction_activity: np.ndarray
    gating_activity: np.ndarray


def run_epoch(
    model: PlasticAttractor,
    inputs: np.ndarray,
    protocol: EpochProtocol,
    *,
    perturbation: ConjunctionClamp | None = None,
    gate_drive: np.ndarray | None = None,
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

    if gate_drive is not None:
        gate_drive = np.asarray(gate_drive, dtype=float)
        expected_gate_shape = (
            protocol.number_of_steps,
            model.number_of_gating_units,
        )
        if gate_drive.shape != expected_gate_shape:
            raise ValueError(
                f'gate_drive must have shape {expected_gate_shape}; '
                f'received {gate_drive.shape}'
            )

    # Activity resets at each epoch. Fast and slow weights keep learning.
    model.reset_activity()

    feature_history = np.empty(
        (protocol.number_of_steps, model.number_of_feature_units)
    )
    conjunction_history = np.empty(
        (protocol.number_of_steps, model.number_of_conjunction_units)
    )
    gating_history = np.empty(
        (protocol.number_of_steps, model.number_of_gating_units)
    )

    for time_step in range(protocol.number_of_steps):
        clamp_is_active = (
            perturbation is not None and perturbation.is_active(time_step)
        )
        state = model.step(
            input_array[time_step],
            clamp_conjunctions=clamp_is_active,
            gate_external_input=(
                None if gate_drive is None else gate_drive[time_step]
            ),
            # True only while the cue is genuinely, externally present. Gates
            # both eligibility-trace accumulation and the gate's own cue-input
            # term (see PlasticAttractor._next_gating_activity) to this same
            # window, on every trial type -- not to whether a forced
            # gate_drive happens to be set, so real trials (which pass no
            # drive) still accumulate from the gate's own learned response,
            # and still rely on its self-sustaining recurrence, not a
            # continuously-reread cue channel, for the rest of the trial.
            cue_signal_active=protocol.stimulus_window.contains(time_step),
            learn=learn,
        )
        feature_history[time_step] = state.feature_activity
        conjunction_history[time_step] = state.conjunction_activity
        gating_history[time_step] = state.gating_activity

    return NetworkTrajectory(feature_history, conjunction_history, gating_history)


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
) -> list[tuple[int, Stimulus]]:
    """Every (cue x stimulus) combo for task, tiled permutation_repeats times + shuffled."""

    combos = [
        (cue, stimulus)
        for cue in cues_for_task(vocabulary, task)
        for stimulus in ALL_STIMULI
    ]
    combos = combos * permutation_repeats
    order = random_generator.permutation(len(combos))
    return [combos[i] for i in order]


def practice_block_plan(
    task: Task,
    permutation_repeats: int,
    vocabulary: Vocabulary,
    random_generator: np.random.RandomState,
) -> list[PlannedTrial]:
    """Plan one practice block: every (cue x stimulus) combo, tiled + shuffled.

    Runs through the full permutation permutation_repeats times, so every cue
    is taught on every stimulus an equal number of times.
    """

    if permutation_repeats <= 0:
        raise ValueError('practice_permutation_repeats must be positive')

    deck = _balanced_shuffled_deck(task, permutation_repeats, vocabulary, random_generator)
    return [PlannedTrial(task, cue, stimulus) for cue, stimulus in deck]


def switching_practice_block_plan(
    switch_probability: float,
    permutation_repeats: int,
    vocabulary: Vocabulary,
    random_generator: np.random.RandomState,
) -> list[PlannedTrial]:
    """Plan one practice block that may switch rule mid-block, like a real block.

    Unlike real_block_plan's i.i.d. cue draws, this keeps practice's balance
    guarantee: one shuffled (cue x stimulus) deck per rule, each covering the
    full permutation permutation_repeats times, so every combo is still taught
    exactly permutation_repeats times per rule even while switching.
    """

    if permutation_repeats <= 0:
        raise ValueError('practice_permutation_repeats must be positive')
    if not 0.0 <= switch_probability <= 1.0:
        raise ValueError('switch_probability must lie between 0 and 1')

    decks = {
        task: _balanced_shuffled_deck(task, permutation_repeats, vocabulary, random_generator)
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


def real_block_plan(
    switch_probability: float,
    num_trials: int,
    vocabulary: Vocabulary,
    random_generator: np.random.RandomState,
    initial_task: Task | None = None,
) -> list[PlannedTrial]:
    """Plan one real block: rule switches w.p. switch_probability, cue signals it.

    num_trials must be a multiple of 4 (four stimuli tiled evenly). The first
    rule is initial_task if given, otherwise random; each later trial switches
    rule with the given probability. The cue is drawn uniformly from the
    active rule's cues, so a rule repeat can still be a cue switch. Stimulus
    identity is independent of the rule.
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

    cues = []
    for task in tasks:
        task_cues = cues_for_task(vocabulary, task)
        cues.append(task_cues[random_generator.randint(len(task_cues))])

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


def _gate_drive_schedule(
    planned: PlannedTrial,
    protocol: EpochProtocol,
    num_gates: int,
) -> np.ndarray:
    """Teaching drive on the rule's gate, held only while the cue is present.

    +1 on the gate for planned.task (PRACTICE_TASKS order), -1 on every other
    gate (mirroring the +-1 action_teach drive). The -1 holds the off-rule gates
    below baseline so their covariance with the current cue is negative: they
    *unlearn* this cue, which is what makes each gate rule-selective.

    Deliberately scoped to stimulus_window, not the much longer
    teaching_window used for the response/irrelevant-pair drives: holding a
    forced +-1 push open past where the cue itself is genuinely, externally
    driven would pair a still-forced gate with an already-silent cue -- two
    units both below baseline read as *positive* covariance under this rule,
    spuriously teaching the off-rule gate to respond to a cue it never saw.
    run_epoch forms the gate eligibility trace over this same stimulus_window
    on every trial (see its cue_signal_active), independently of whether this
    forced drive is present, so the trace/drive alignment holds by
    construction rather than by this function's choice of window. Real trials
    pass no forced drive at all -- the gate fires from its learned cue input
    alone -- but still accumulate and consolidate a trace from that unforced
    response. Shape is (steps, num_gates).
    """

    gate_index = PRACTICE_TASKS.index(planned.task)
    per_gate = np.where(np.arange(num_gates) == gate_index, 1.0, -1.0)
    drive = np.zeros((protocol.number_of_steps, num_gates))
    window = protocol.stimulus_window
    drive[window.start : window.stop] = per_gate
    return drive


def _gate_winner_matches_task(
    trajectory: NetworkTrajectory,
    protocol: EpochProtocol,
    task: Task,
) -> bool:
    """Did the gate with the higher mean activity over stimulus_window match task?

    The reward signal for gate consolidation: whether the *gate itself* picked
    the true rule, not whether the network's overall response was correct.
    Overall correctness also depends on the main W conjunction mapping,
    congruency, and switch costs -- none of which the gate can be blamed or
    credited for -- so using it as the gate's reward let unrelated errors
    depress an already-correct gate mapping (and unrelated correct guesses
    reinforce a wrong one). During instruction the gate is externally forced
    onto the correct side (see _gate_drive_schedule), so its own winner
    matches task there by construction too: this is one reward rule for every
    trial type, not a special case for instruction.
    """

    window = protocol.stimulus_window
    gate_mean = trajectory.gating_activity[window.start : window.stop].mean(axis=0)
    winner = int(np.argmax(gate_mean))
    return winner == PRACTICE_TASKS.index(task)


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
    # the task-irrelevant colour/shape pair is also overridden here, to a
    # neutral .5 rather than the +-1 used for the response units, for the same
    # window. this does NOT come from attractor_rnn.py (the cued flat script
    # has no such override anywhere); it's carried over from the separate,
    # pre-stimulus rule-cue instruction used by the non-cued published model
    # (published_model/plastic_attractor/experiment.py's rule vector), kept
    # here as a deliberate local addition to the cued teaching drive.
    if apply_teaching:
        correct = correct_response(vocabulary, planned.task, planned.stimulus)
        teaching_window = protocol.teaching_window
        inputs[
            teaching_window.start : teaching_window.stop,
            list(vocabulary.response_features),
        ] = _teaching_drive(correct, vocabulary)
        inputs[
            teaching_window.start : teaching_window.stop,
            list(IRRELEVANT_FEATURES_BY_TASK[planned.task]),
        ] = 0.5

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
    # gate teaching drive only on instruction trials; real trials leave the gate
    # to fire from its learned cue input. None when gating is disabled.
    gate_drive = None
    if model.number_of_gating_units > 0 and apply_teaching:
        gate_drive = _gate_drive_schedule(
            planned, config.protocol, model.number_of_gating_units
        )

    trajectory = run_epoch(
        model,
        _trial_epoch(planned, apply_teaching, config.protocol, vocabulary),
        config.protocol,
        perturbation=config.perturbation,
        gate_drive=gate_drive,
        learn=config.learn_during_trials,
    )
    chosen_response, reaction_time = _measure_response(
        trajectory,
        config.protocol,
        vocabulary,
    )

    # Reward-gated consolidation of the gate eligibility trace (three-factor
    # rule), every trial: +1 potentiates what the trial's coincident activity
    # accumulated, -1 applies the same trace with a flipped sign. Reward comes
    # from _gate_winner_matches_task (did the gate itself pick the true rule),
    # not overall response correctness -- an earlier version used response
    # correctness and it consolidated cleanly during instruction (always
    # correct there, response forced) but derailed within the first
    # post-instruction block on every seed tested: an incorrect response
    # caused by the main W conjunction mapping, congruency, or a switch cost
    # has nothing to do with the gate, and punishing the gate for it eroded an
    # already-correct mapping. This relies on the trace itself only ever
    # accumulating during stimulus_window (see step()'s cue_signal_active),
    # where the cue is genuinely, currently active rather than decayed back to
    # baseline -- that's what makes it safe to consolidate real-trial activity
    # at all; consolidating a trace built from a stale/silent cue was the
    # original derailment bug. No-op when gating is disabled.
    ground_truth = correct_response(vocabulary, planned.task, planned.stimulus)
    if model.number_of_gating_units > 0:
        gate_is_correct = _gate_winner_matches_task(trajectory, config.protocol, planned.task)
        model.consolidate_gating_trace(1.0 if gate_is_correct else -1.0)

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
        correct_response=ground_truth,
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
    for block_index in range(config.num_practice_blocks):
        if config.practice_switch_probability > 0.0:
            plan = switching_practice_block_plan(
                config.practice_switch_probability,
                config.practice_permutation_repeats,
                vocabulary, random_generator,
            )
        else:
            plan = practice_block_plan(
                PRACTICE_TASKS[block_index], config.practice_permutation_repeats,
                vocabulary, random_generator,
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
    # learning stays on (learn_during_trials), so this is still training.
    if config.include_performance_practice:
        for practice_index, task in enumerate(PRACTICE_TASKS):
            block_index = config.num_practice_blocks + practice_index
            plan = real_block_plan(
                switch_probability=0.0,
                num_trials=config.num_trials,
                vocabulary=vocabulary,
                random_generator=random_generator,
                initial_task=task,
            )
            trials.extend(
                _run_block(
                    model, config, vocabulary, block_index,
                    is_practice=True, apply_teaching=False,
                    switch_probability=None, plan=plan,
                )
            )
            eigenvalue_counts.append(model.number_of_amplifying_eigenvalues())

    # real blocks: no teaching, the cue signals the rule that varies trial by
    # trial. learning stays on so the weights keep evolving.
    performance_practice_blocks = (
        len(PRACTICE_TASKS) if config.include_performance_practice else 0
    )
    for real_index, switch_probability in enumerate(config.switch_probs):
        block_index = (
            config.num_practice_blocks + performance_practice_blocks + real_index
        )
        plan = real_block_plan(
            switch_probability, config.num_trials, vocabulary, random_generator
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
