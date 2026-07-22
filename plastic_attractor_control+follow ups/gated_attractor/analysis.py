"""Behavioural and eigenvalue summaries calculated from one experiment.

Every function here works on a single ExperimentResult (one seed). Averaging
across seeds (mean +/- s.e.m.) is left to the launcher, which stacks these
per-seed outputs. This mirrors how the flat simulation_launcher.py computed
things, but without the trial x block x seed index arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean

import numpy as np

from .experiment import (
    ExperimentResult,
    TransitionType,
    TrialResult,
    gate_winner_matches_task,
)
from .task import Feature, Task


@dataclass(frozen=True)
class BehavioralSummary:
    """The main accuracy and reaction-time measurements."""

    accuracy: float
    mean_reaction_time_in_steps: float
    congruent_reaction_time_in_steps: float
    incongruent_reaction_time_in_steps: float

    @property
    def congruency_effect_in_steps(self) -> float:
        """Positive values mean incongruent trials took longer."""

        return (
            self.incongruent_reaction_time_in_steps
            - self.congruent_reaction_time_in_steps
        )

    def as_dictionary(self) -> dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'mean_reaction_time_in_steps': self.mean_reaction_time_in_steps,
            'congruent_reaction_time_in_steps': (
                self.congruent_reaction_time_in_steps
            ),
            'incongruent_reaction_time_in_steps': (
                self.incongruent_reaction_time_in_steps
            ),
            'congruency_effect_in_steps': self.congruency_effect_in_steps,
        }


@dataclass(frozen=True)
class Contrast:
    """A hard-vs-easy comparison (switch-vs-repeat or incongruent-vs-congruent).

    Costs are signed so a positive number always means the harder condition was
    worse: rt_cost = hard - easy RT, accuracy_cost = easy - hard accuracy.
    """

    rt_cost: float
    accuracy_cost: float
    hard_rt: float
    easy_rt: float
    hard_accuracy: float
    easy_accuracy: float


def _mean_or_nan(values: list[float]) -> float:
    return fmean(values) if values else float('nan')


def summarize_behavior(trials: Iterable[TrialResult]) -> BehavioralSummary:
    """Calculate accuracy and correct-trial reaction-time summaries."""

    trial_list = list(trials)
    if not trial_list:
        raise ValueError('At least one trial is required')

    # Reaction time is defined only for trials with the correct response.
    correct_trials = [trial for trial in trial_list if trial.correct]
    congruent_times = [
        trial.reaction_time_in_steps
        for trial in correct_trials
        if trial.congruent
    ]
    incongruent_times = [
        trial.reaction_time_in_steps
        for trial in correct_trials
        if not trial.congruent
    ]

    return BehavioralSummary(
        accuracy=sum(trial.correct for trial in trial_list) / len(trial_list),
        mean_reaction_time_in_steps=_mean_or_nan(
            [trial.reaction_time_in_steps for trial in correct_trials]
        ),
        congruent_reaction_time_in_steps=_mean_or_nan(congruent_times),
        incongruent_reaction_time_in_steps=_mean_or_nan(incongruent_times),
    )


def accuracy_by_task(trials: Iterable[TrialResult]) -> dict[Task, float]:
    """Accuracy split by which rule was active on the trial.

    An aggregate accuracy number (summarize_behavior) can look fine while one
    rule is carrying it and the other is near chance -- with only 4
    conjunction units for 2 rules x 2 responses, that asymmetry is exactly
    the kind of failure a single pooled number hides. NaN for a task with no
    trials in the group.
    """

    trial_list = list(trials)
    return {
        task: _mean_or_nan(
            [float(trial.correct) for trial in trial_list if trial.task == task]
        )
        for task in Task
    }


# EIGENVALUE HELPERS #
# ===================================================== #

def _amplifying_count(weights: np.ndarray) -> int:
    """Number of eigenvalues of W W.T greater than one (amplifying modes)."""

    eigenvalues = np.linalg.eigvalsh(weights @ weights.T)
    return int(np.sum(eigenvalues > 1.0))


def _eigenvalue_magnitudes(weights: np.ndarray) -> np.ndarray:
    """Magnitude of every eigenvalue of W W.T (one per feature unit)."""

    return np.abs(np.linalg.eigvalsh(weights @ weights.T))


def practice_learning_curve(result: ExperimentResult) -> np.ndarray:
    """Amplifying-eigenvalue count per instruction trial, block 0 then block 1.

    One continuous curve of length num_practice_blocks x practice_trials. Trials
    are already stored in block-then-trial order, so filtering the instruction
    trials (the ones with the artificial teaching drive, as opposed to the
    later performance-practice blocks) preserves the concatenation the plot
    expects.
    """

    return np.array(
        [
            _amplifying_count(trial.combined_weights)
            for trial in result.trials
            if trial.is_instruction
        ]
    )


def block_kinds(result: ExperimentResult) -> list[float]:
    """The distinct switch probabilities used across the real blocks, sorted."""

    return sorted(set(result.config.switch_probs))


def _real_trials_by_kind(
    result: ExperimentResult,
) -> dict[float, list[TrialResult]]:
    """Group real (non-practice) trials by their block's switch probability."""

    grouped: dict[float, list[TrialResult]] = {
        kind: [] for kind in block_kinds(result)
    }
    for trial in result.trials:
        if not trial.is_practice:
            grouped[trial.switch_probability].append(trial)
    return grouped


def amplifying_eigenvalue_mean_by_kind(
    result: ExperimentResult,
) -> dict[float, float]:
    """Mean amplifying-eigenvalue count over each block kind's real trials."""

    return {
        kind: _mean_or_nan(
            [_amplifying_count(trial.combined_weights) for trial in trials]
        )
        for kind, trials in _real_trials_by_kind(result).items()
    }


def eigenvalue_magnitudes_by_kind(
    result: ExperimentResult,
) -> dict[float, np.ndarray]:
    """Pooled eigenvalue magnitudes over each block kind's real trials.

    Flattens across trials and feature units, for the by-kind spread boxplot.
    """

    return {
        kind: np.concatenate(
            [_eigenvalue_magnitudes(trial.combined_weights) for trial in trials]
        )
        if trials
        else np.array([])
        for kind, trials in _real_trials_by_kind(result).items()
    }


# BEHAVIOURAL CONTRASTS #
# ===================================================== #

def _contrast(
    trials: list[TrialResult],
    is_hard,
    is_easy,
) -> Contrast:
    """RT and accuracy comparison between a hard and an easy trial group.

    RT uses correct trials only; accuracy uses every trial of the group.
    """

    hard = [trial for trial in trials if is_hard(trial)]
    easy = [trial for trial in trials if is_easy(trial)]

    hard_rt = _mean_or_nan(
        [trial.reaction_time_in_steps for trial in hard if trial.correct]
    )
    easy_rt = _mean_or_nan(
        [trial.reaction_time_in_steps for trial in easy if trial.correct]
    )
    hard_accuracy = _mean_or_nan([float(trial.correct) for trial in hard])
    easy_accuracy = _mean_or_nan([float(trial.correct) for trial in easy])

    return Contrast(
        rt_cost=hard_rt - easy_rt,
        accuracy_cost=easy_accuracy - hard_accuracy,
        hard_rt=hard_rt,
        easy_rt=easy_rt,
        hard_accuracy=hard_accuracy,
        easy_accuracy=easy_accuracy,
    )


def switch_contrast_by_kind(result: ExperimentResult) -> dict[float, Contrast]:
    """Rule-switch vs full-repeat contrast per block kind.

    Cue-switch/rule-repeat trials are excluded from both groups so the contrast
    isolates rule switching (hard = RULE_SWITCH, easy = CUE_REPEAT).
    """

    return {
        kind: _contrast(
            trials,
            is_hard=lambda t: t.transition_type == TransitionType.RULE_SWITCH,
            is_easy=lambda t: t.transition_type == TransitionType.CUE_REPEAT,
        )
        for kind, trials in _real_trials_by_kind(result).items()
    }


def incongruence_contrast_by_kind(
    result: ExperimentResult,
) -> dict[float, Contrast]:
    """Incongruent vs congruent contrast per block kind."""

    return {
        kind: _contrast(
            trials,
            is_hard=lambda t: not t.congruent,
            is_easy=lambda t: t.congruent,
        )
        for kind, trials in _real_trials_by_kind(result).items()
    }


# EXPERIMENTAL CONTROLS (novel cue / omitted combo) #
# ===================================================== #
#
# Controls 2 and 3 (see SwitchingExperimentConfig.practice_cue_restriction /
# omit_practice_combo) only change what's taught; the real (test) blocks
# always include every cue and every combo. The functions below classify
# real trials against the config that produced them, one ExperimentResult
# (one seed) at a time, consistent with the rest of this module -- stacking
# across seeds is left to the caller, as with everything else here.
#
# The network (and the gate, whose weights only update from cues it actually
# sees) keeps learning during testing, so a novel cue or omitted combo stops
# being a genuine transfer test after its first few occurrences at test.
# novel_cue_exposure_sequence and omitted_combo_exposure_sequence return the
# matching real trials in the order the network actually encountered them,
# so a caller can look at accuracy by exposure number instead of only a
# single "first vs rest" split.

def _is_novel_cue(trial: TrialResult, restriction: dict) -> bool:
    trained = restriction.get(trial.task)
    return trained is not None and trial.cue not in trained


def _is_trained_cue(trial: TrialResult, restriction: dict) -> bool:
    trained = restriction.get(trial.task)
    return trained is not None and trial.cue in trained


def novel_cue_contrast(result: ExperimentResult) -> Contrast:
    """Novel-cue vs trained-cue contrast (Control 2) among real trials.

    Requires config.practice_cue_restriction. 'Novel' means the cue was
    excluded from every practice / performance-practice block for its rule;
    'trained' means it was the (only) cue taught for that rule.
    """

    restriction = result.config.practice_cue_restriction
    if restriction is None:
        raise ValueError('novel_cue_contrast requires practice_cue_restriction')

    real_trials = [trial for trial in result.trials if not trial.is_practice]
    return _contrast(
        real_trials,
        is_hard=lambda t: _is_novel_cue(t, restriction),
        is_easy=lambda t: _is_trained_cue(t, restriction),
    )


def novel_cue_exposure_sequence(result: ExperimentResult) -> list[TrialResult]:
    """Real trials on a novel cue (Control 2), in encounter order.

    Requires config.practice_cue_restriction.
    """

    restriction = result.config.practice_cue_restriction
    if restriction is None:
        raise ValueError('novel_cue_exposure_sequence requires practice_cue_restriction')

    return [
        trial for trial in result.trials
        if not trial.is_practice and _is_novel_cue(trial, restriction)
    ]


def omitted_combo_contrast(result: ExperimentResult) -> Contrast:
    """Omitted-combo vs other-combo contrast (Control 3) among real trials.

    Requires config.omit_practice_combo.
    """

    omit_combo = result.config.omit_practice_combo
    if omit_combo is None:
        raise ValueError('omitted_combo_contrast requires omit_practice_combo')

    real_trials = [trial for trial in result.trials if not trial.is_practice]
    return _contrast(
        real_trials,
        is_hard=lambda t: (t.cue, t.stimulus) == omit_combo,
        is_easy=lambda t: (t.cue, t.stimulus) != omit_combo,
    )


def omitted_combo_exposure_sequence(result: ExperimentResult) -> list[TrialResult]:
    """Real trials reproducing the omitted (cue, stimulus) combo (Control 3),
    in encounter order. Requires config.omit_practice_combo.
    """

    omit_combo = result.config.omit_practice_combo
    if omit_combo is None:
        raise ValueError('omitted_combo_exposure_sequence requires omit_practice_combo')

    return [
        trial for trial in result.trials
        if not trial.is_practice and (trial.cue, trial.stimulus) == omit_combo
    ]


def accuracy_by_task_by_kind(
    result: ExperimentResult,
) -> dict[float, dict[Task, float]]:
    """accuracy_by_task, further split by block kind (switch probability)."""

    return {
        kind: accuracy_by_task(trials)
        for kind, trials in _real_trials_by_kind(result).items()
    }


# PERFORMANCE ACROSS BLOCKS #
# ===================================================== #

def performance_by_block(
    result: ExperimentResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean RT (correct trials) and accuracy for every block, in block order.

    Covers practice and real blocks together, to show how performance evolves
    over the whole session. Returns (rt_by_block, accuracy_by_block).
    """

    number_of_blocks = result.config.number_of_blocks
    reaction_time = np.full(number_of_blocks, np.nan)
    accuracy = np.full(number_of_blocks, np.nan)

    for block_index in range(number_of_blocks):
        block_trials = [
            trial for trial in result.trials if trial.block_index == block_index
        ]
        if not block_trials:
            continue
        reaction_time[block_index] = _mean_or_nan(
            [
                trial.reaction_time_in_steps
                for trial in block_trials
                if trial.correct
            ]
        )
        accuracy[block_index] = _mean_or_nan(
            [float(trial.correct) for trial in block_trials]
        )

    return reaction_time, accuracy


# ROOT-CAUSE DIAGNOSTICS BY BLOCK #
# ===================================================== #

def gate_accuracy_by_block(result: ExperimentResult) -> np.ndarray:
    """Fraction of trials per block where the gate's winner matched the task.

    NaN where gating is disabled (nothing to measure) or a block has no
    trials, mirroring performance_by_block's block-index convention.
    """

    number_of_blocks = result.config.number_of_blocks
    accuracy = np.full(number_of_blocks, np.nan)
    if result.config.model_parameters.number_of_gating_units == 0:
        return accuracy

    protocol = result.config.protocol
    for block_index in range(number_of_blocks):
        block_trials = [
            trial for trial in result.trials if trial.block_index == block_index
        ]
        if not block_trials:
            continue
        accuracy[block_index] = _mean_or_nan([
            float(gate_winner_matches_task(trial.trajectory, protocol, trial.task))
            for trial in block_trials
        ])

    return accuracy


def no_response_rate_by_block(result: ExperimentResult) -> np.ndarray:
    """Fraction of trials per block with no measurable response
    (chosen_response is None), mirroring performance_by_block's convention.
    """

    number_of_blocks = result.config.number_of_blocks
    rate = np.full(number_of_blocks, np.nan)

    for block_index in range(number_of_blocks):
        block_trials = [
            trial for trial in result.trials if trial.block_index == block_index
        ]
        if not block_trials:
            continue
        rate[block_index] = _mean_or_nan(
            [float(trial.chosen_response is None) for trial in block_trials]
        )

    return rate


# colour/shape rows of W, in a fixed order every diagnostic below shares
_COLOUR_SHAPE_FEATURES = (Feature.GREEN, Feature.BLUE, Feature.SQUARE, Feature.CIRCLE)


def conjunction_unit_discrimination(weights: np.ndarray) -> np.ndarray:
    """Per-conjunction-unit within-dimension weight gap: how well each unit,
    on its own, tells green from blue and square from circle.

    Shape (num_conjunction_units, 2), columns [colour_gap, shape_gap]. Earlier
    ad hoc checks of this session's plasticity-pause fix looked only at the
    *best* (max) gap across units, which can improve even while other units
    stay collapsed or duplicated -- see model_outline.md's open questions.
    Whyte et al.'s own criterion needs >=2 *simultaneously* well-differentiated
    units (one per genuinely distinct S-R mapping; congruent pairs may share
    one), so the full per-unit array, not its max, is what should be
    inspected from now on.
    """

    green, blue, square, circle = (
        weights[int(Feature.GREEN)], weights[int(Feature.BLUE)],
        weights[int(Feature.SQUARE)], weights[int(Feature.CIRCLE)],
    )
    return np.stack([np.abs(green - blue), np.abs(square - circle)], axis=1)


def conjunction_unit_discrimination_by_block(result: ExperimentResult) -> np.ndarray:
    """conjunction_unit_discrimination for one snapshot (the block's last
    trial) per block, mirroring colour_shape_row_norms_by_block. Shape
    (num_blocks, num_conjunction_units, 2).
    """

    number_of_blocks = result.config.number_of_blocks
    num_units = result.config.model_parameters.number_of_conjunction_units
    gaps = np.full((number_of_blocks, num_units, 2), np.nan)

    for block_index in range(number_of_blocks):
        block_trials = [
            trial for trial in result.trials if trial.block_index == block_index
        ]
        if not block_trials:
            continue
        gaps[block_index] = conjunction_unit_discrimination(
            block_trials[-1].combined_weights
        )

    return gaps


def colour_shape_row_norms_by_block(result: ExperimentResult) -> np.ndarray:
    """L2 norm of W's green/blue/square/circle rows, one snapshot (the
    block's last trial) per block. Shape (num_blocks, 4).
    """

    number_of_blocks = result.config.number_of_blocks
    rows = np.array([int(feature) for feature in _COLOUR_SHAPE_FEATURES])
    norms = np.full((number_of_blocks, rows.size), np.nan)

    for block_index in range(number_of_blocks):
        block_trials = [
            trial for trial in result.trials if trial.block_index == block_index
        ]
        if not block_trials:
            continue
        norms[block_index] = np.linalg.norm(
            block_trials[-1].combined_weights[rows], axis=1
        )

    return norms


# ROOT-CAUSE DIAGNOSTICS BY BLOCK KIND #
# ===================================================== #

@dataclass(frozen=True)
class ActivityLevels:
    """Settled feature activity, split by whether the current rule cares
    about it."""

    relevant: float
    irrelevant: float


# how many steps at the end of response_window count as "settled", not the
# transient right after the stimulus/cue turn off
_SETTLED_ACTIVITY_WINDOW_STEPS = 20


def relevant_irrelevant_activity_by_kind(
    result: ExperimentResult,
) -> dict[float, ActivityLevels]:
    """Mean settled (relevant, irrelevant) feature activity over each block
    kind's real trials, averaged over response_window's last
    _SETTLED_ACTIVITY_WINDOW_STEPS steps.
    """

    window = result.config.protocol.response_window
    start = max(window.start, window.stop - _SETTLED_ACTIVITY_WINDOW_STEPS)

    levels: dict[float, ActivityLevels] = {}
    for kind, trials in _real_trials_by_kind(result).items():
        relevant_values = []
        irrelevant_values = []
        for trial in trials:
            activity = trial.trajectory.feature_activity[start : window.stop]
            relevant_values.append(
                float(activity[:, trial.stimulus.relevant_feature(trial.task)].mean())
            )
            irrelevant_values.append(
                float(activity[:, trial.stimulus.irrelevant_feature(trial.task)].mean())
            )
        levels[kind] = ActivityLevels(
            relevant=_mean_or_nan(relevant_values),
            irrelevant=_mean_or_nan(irrelevant_values),
        )

    return levels


# ROUTING DIAGNOSTICS #
# ===================================================== #

def _settled_conjunction_winner(trial: TrialResult, window) -> int:
    """Which conjunction unit holds the most settled activity, averaged over
    the last _SETTLED_ACTIVITY_WINDOW_STEPS of response_window -- the same
    settled window relevant_irrelevant_activity_by_kind uses."""

    start = max(window.start, window.stop - _SETTLED_ACTIVITY_WINDOW_STEPS)
    settled = trial.trajectory.conjunction_activity[start : window.stop]
    return int(np.argmax(settled.mean(axis=0)))


def _routing_flip_rate(trials: Iterable[TrialResult], window) -> float:
    """Fraction of (task, stimulus) identities repeated at least twice among
    trials whose settled winner-take-all conjunction unit isn't the same on
    every occurrence. NaN if no identity repeats."""

    winners_by_stimulus: dict = {}
    for trial in trials:
        key = (trial.task, trial.stimulus)
        winners_by_stimulus.setdefault(key, []).append(
            _settled_conjunction_winner(trial, window)
        )
    repeated = [winners for winners in winners_by_stimulus.values() if len(winners) >= 2]
    return _mean_or_nan([float(len(set(winners)) > 1) for winners in repeated])


def conjunction_routing_flip_rate_by_block(result: ExperimentResult) -> np.ndarray:
    """Within-block routing stability: for each block, the fraction of
    repeated same-stimulus presentations whose settled winner-take-all
    conjunction unit changes from one presentation to another.

    NaN for a block with no stimulus repeated at least twice, mirroring
    performance_by_block's block-index convention. See model_outline.md
    section 13 -- the winning unit for a fixed physical stimulus is not
    guaranteed to stay the same once real-block learning stays on.
    """

    number_of_blocks = result.config.number_of_blocks
    window = result.config.protocol.response_window
    rate = np.full(number_of_blocks, np.nan)

    for block_index in range(number_of_blocks):
        block_trials = [
            trial for trial in result.trials if trial.block_index == block_index
        ]
        if block_trials:
            rate[block_index] = _routing_flip_rate(block_trials, window)

    return rate


def conjunction_routing_drift_by_kind(result: ExperimentResult) -> dict[float, float]:
    """Cross-block routing stability: for each block kind (switch
    probability), the fraction of (task, stimulus) identities whose settled
    winner-take-all conjunction unit isn't the same across every real trial
    of that kind (typically 2 blocks). NaN for a kind with no stimulus
    repeated at least twice. Complements conjunction_routing_flip_rate_by_
    block, which asks the same question within a single block instead.
    """

    window = result.config.protocol.response_window
    return {
        kind: _routing_flip_rate(trials, window)
        for kind, trials in _real_trials_by_kind(result).items()
    }


def conjunction_routing_flip_rate_full_session(result: ExperimentResult) -> float:
    """Whole-session routing stability: fraction of (task, stimulus)
    identities whose settled winner-take-all conjunction unit isn't the same
    across every real trial, pooling all real blocks together (not just the
    ~2 sharing a switch-probability kind, as conjunction_routing_drift_by_kind
    does). NaN if no identity repeats. Directly comparable to
    model_outline.md section 13's whole-session "0/96" headline number.
    """

    window = result.config.protocol.response_window
    all_real_trials = [
        trial for trials in _real_trials_by_kind(result).values() for trial in trials
    ]
    return _routing_flip_rate(all_real_trials, window)
