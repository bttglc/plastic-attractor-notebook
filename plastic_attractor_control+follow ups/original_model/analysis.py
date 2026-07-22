"""Small behavioral summaries calculated from trial results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean

from .experiment import ExperimentResult, TrialResult
from .task import Stimulus, Task


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
            "accuracy": self.accuracy,
            "mean_reaction_time_in_steps": self.mean_reaction_time_in_steps,
            "congruent_reaction_time_in_steps": (
                self.congruent_reaction_time_in_steps
            ),
            "incongruent_reaction_time_in_steps": (
                self.incongruent_reaction_time_in_steps
            ),
            "congruency_effect_in_steps": self.congruency_effect_in_steps,
        }


def _mean_or_nan(values: list[float]) -> float:
    return fmean(values) if values else float("nan")


def summarize_behavior(trials: Iterable[TrialResult]) -> BehavioralSummary:
    """Calculate accuracy and correct-trial reaction-time summaries."""

    trial_list = list(trials)
    if not trial_list:
        raise ValueError("At least one trial is required")

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


# OMITTED-COMBO CONTROL #
# ===================================================== #

@dataclass(frozen=True)
class Contrast:
    """A hard-vs-easy comparison. Costs are signed so a positive number
    always means the harder condition was worse."""

    rt_cost: float
    accuracy_cost: float
    hard_rt: float
    easy_rt: float
    hard_accuracy: float
    easy_accuracy: float


def _contrast(trials, is_hard, is_easy) -> Contrast:
    """RT and accuracy comparison between a hard and an easy trial group.

    RT uses correct trials only; accuracy uses every trial of the group.
    """

    hard = [t for t in trials if is_hard(t)]
    easy = [t for t in trials if is_easy(t)]

    hard_rt = _mean_or_nan([t.reaction_time_in_steps for t in hard if t.correct])
    easy_rt = _mean_or_nan([t.reaction_time_in_steps for t in easy if t.correct])
    hard_accuracy = _mean_or_nan([float(t.correct) for t in hard])
    easy_accuracy = _mean_or_nan([float(t.correct) for t in easy])

    return Contrast(
        rt_cost=hard_rt - easy_rt, accuracy_cost=easy_accuracy - hard_accuracy,
        hard_rt=hard_rt, easy_rt=easy_rt, hard_accuracy=hard_accuracy, easy_accuracy=easy_accuracy,
    )


def _is_omitted_combo(trial: TrialResult, omit_combo: tuple[Task, Stimulus]) -> bool:
    return trial.task == omit_combo[0] and trial.stimulus == omit_combo[1]


def omitted_combo_contrast(result: ExperimentResult) -> Contrast:
    """Omitted-combo vs every-other-trial contrast (adapted Control 3).

    Requires config.omit_combo. Trials before omit_combo_until_block never
    match the omitted combo by construction, so they fall entirely into the
    'easy' (other) group -- no separate block filtering needed.
    """

    omit_combo = result.config.omit_combo
    if omit_combo is None:
        raise ValueError('omitted_combo_contrast requires config.omit_combo')

    return _contrast(
        result.trials,
        is_hard=lambda t: _is_omitted_combo(t, omit_combo),
        is_easy=lambda t: not _is_omitted_combo(t, omit_combo),
    )


def omitted_combo_exposure_sequence(result: ExperimentResult) -> list[TrialResult]:
    """Trials reproducing the omitted (task, stimulus) combo, in encounter
    order. Requires config.omit_combo. Empty until omit_combo_until_block.
    """

    omit_combo = result.config.omit_combo
    if omit_combo is None:
        raise ValueError('omitted_combo_exposure_sequence requires config.omit_combo')

    return [t for t in result.trials if _is_omitted_combo(t, omit_combo)]
