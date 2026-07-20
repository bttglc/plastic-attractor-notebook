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

from .experiment import ExperimentResult, TransitionType, TrialResult


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


# RESPONSE DIAGNOSTICS #
# ===================================================== #

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


# ROUTING DIAGNOSTICS #
# ===================================================== #

# how many steps at the end of response_window count as "settled", not the
# transient right after the stimulus/cue turn off (matches the sibling
# packages' relevant_irrelevant_activity_by_kind convention)
_SETTLED_ACTIVITY_WINDOW_STEPS = 20


def _settled_conjunction_winner(trial: TrialResult, window) -> int:
    """Which conjunction unit holds the most settled activity, averaged over
    the last _SETTLED_ACTIVITY_WINDOW_STEPS of response_window."""

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
    performance_by_block's block-index convention. gated_attractor's
    model_outline.md (section 13) found the winning unit for a fixed
    physical stimulus is not guaranteed to stay the same once real-block
    learning stays on -- tracked here for comparison across updated_model
    variants, including this cue-only, no-gating baseline.
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
    gated_attractor's model_outline.md section 13 whole-session "0/96"
    headline number.
    """

    window = result.config.protocol.response_window
    all_real_trials = [
        trial for trials in _real_trials_by_kind(result).values() for trial in trials
    ]
    return _routing_flip_rate(all_real_trials, window)
