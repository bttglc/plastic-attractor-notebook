"""Port of the published model's Figure-3 decoding analysis to gated_attractor.

The original decoder (copy_of_published_model_explained_step_by_step.py,
`decode_dimension_over_time` / `decode_by_relevance`) trains a held-out-block,
cross-validated logistic regression on population activity at every trial
time step, separately for the task-relevant and task-irrelevant stimulus
dimension. This reuses that exact method -- same solver, same balanced
accuracy, same block-held-out folds -- adapted for gated_attractor's trial
format. Three real differences from the original, each noted where it bites:

1. Real vs. practice/instruction trials. The original model's trial pool
   doesn't carry that distinction the same way; gated_attractor's does
   (`TrialResult.is_practice`), and every other analysis in this project
   (accuracy, eigenvalues, routing drift) restricts to real trials only --
   see `_real_trials_by_kind` in gated_attractor/analysis.py. The decoder
   filter below does the same, so it's not measuring the instruction-window
   teaching drive by accident.

2. Feature indexing. The original slices `SENSORY_FEATURES` out of a
   feature_activity array that also holds motor units. gated_attractor's
   colour/shape units sit at fixed indices 0-3 regardless of how many cues
   the vocabulary has (see task.py's Vocabulary docstring), so the slice is
   simpler here, not different in kind.

3. Task assignment. The original alternates whole blocks between COLOR and
   SHAPE, so held-out-block cross-validation means "generalise to an unseen
   block of the same task." gated_attractor assigns task per trial via the
   cue, so a real block can contain both tasks -- held-out-block folds still
   prevent same-block leakage, they just no longer mean "unseen block of
   this task specifically." Worth saying out loud if this gets presented.

Run it from the project root:

    python decode_gated_attractor.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.getcwd()
sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.join(_this_dir, 'gated_attractor'))

from gated_attractor import SwitchingExperimentConfig, Task, run_switching_experiment
from gated_attractor.task import COLOR_FEATURES, Feature, SHAPE_FEATURES
from model_versions_config import model_versions

# same run config as gated_attractor/run_controls.py's baseline condition
num_trials = 48
practice_permutation_repeats = 5
switch_probs = (0.125, 0.25, 0.5, 0.75) * 2
model_parameters = model_versions['2cpr_slowW3']

SENSORY_FEATURE_INDICES = tuple(
    int(feature) for feature in (COLOR_FEATURES + SHAPE_FEATURES)
)


def _stimulus_label(trial, dimension: Task) -> int:
    """0/1 label for the requested stimulus dimension, same convention as
    the original: green/square -> 0, blue/circle -> 1."""

    if dimension == Task.COLOR:
        return int(trial.stimulus.color == Feature.BLUE)
    return int(trial.stimulus.shape == Feature.CIRCLE)


def _population_activity(trial, population: str) -> np.ndarray:
    """Time-by-unit activity for one population, on one trial."""

    if population == 'feature':
        return trial.trajectory.feature_activity[:, list(SENSORY_FEATURE_INDICES)]
    if population == 'conjunction':
        return trial.trajectory.conjunction_activity
    raise ValueError("population must be 'feature' or 'conjunction'")


def decode_dimension_over_time(
    trials,
    *,
    task: Task,
    decoded_dimension: Task,
    population: str,
    time_steps: np.ndarray,
    number_of_folds: int = 5,
) -> np.ndarray:
    """Decode one stimulus dimension within one task context, at every
    requested time step, using block-held-out cross-validated logistic
    regression. Direct port of the original model's decoder."""

    time_step_values = np.asarray(time_steps, dtype=int)

    # real, correct trials from the requested task only -- see difference
    # (1) in the module docstring: this is the gated_attractor-specific
    # filter the original trial pool didn't need.
    selected_trials = [
        trial for trial in trials
        if trial.correct and trial.task == task and not trial.is_practice
    ]
    if not selected_trials:
        return np.full(time_step_values.size, np.nan)

    labels = np.array(
        [_stimulus_label(trial, decoded_dimension) for trial in selected_trials]
    )
    groups = np.array([trial.block_index for trial in selected_trials])
    activity = np.stack(
        [_population_activity(trial, population) for trial in selected_trials]
    )

    if np.unique(groups).size < number_of_folds or np.unique(labels).size != 2:
        return np.full(time_step_values.size, np.nan)

    splitter = GroupKFold(n_splits=number_of_folds)
    folds = list(splitter.split(activity, labels, groups))
    accuracy = np.empty(time_step_values.size, dtype=float)

    for time_index, time_step in enumerate(time_step_values):
        fold_scores = []
        for train_indices, test_indices in folds:
            train_labels = labels[train_indices]
            test_labels = labels[test_indices]
            if np.unique(train_labels).size < 2 or np.unique(test_labels).size < 2:
                continue

            decoder = LogisticRegression(
                solver='lbfgs', class_weight='balanced', max_iter=1000,
            )
            decoder.fit(activity[train_indices, time_step], train_labels)
            predictions = decoder.predict(activity[test_indices, time_step])
            fold_scores.append(balanced_accuracy_score(test_labels, predictions))

        accuracy[time_index] = float(np.mean(fold_scores)) if fold_scores else float('nan')

    return accuracy


def decode_by_relevance(
    trials,
    *,
    population: str,
    time_steps: np.ndarray,
    number_of_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Task-relevant and task-irrelevant decoding curves, averaged across
    the COLOR and SHAPE task contexts. Unchanged from the original -- the
    task-generic logic doesn't need to know anything about gated_attractor."""

    relevant_curves, irrelevant_curves = [], []

    for task in Task:
        irrelevant_dimension = Task.SHAPE if task == Task.COLOR else Task.COLOR

        relevant_curves.append(
            decode_dimension_over_time(
                trials, task=task, decoded_dimension=task,
                population=population, time_steps=time_steps,
                number_of_folds=number_of_folds,
            )
        )
        irrelevant_curves.append(
            decode_dimension_over_time(
                trials, task=task, decoded_dimension=irrelevant_dimension,
                population=population, time_steps=time_steps,
                number_of_folds=number_of_folds,
            )
        )

    return (
        np.nanmean(relevant_curves, axis=0),
        np.nanmean(irrelevant_curves, axis=0),
    )


if __name__ == '__main__':
    print('Running one gated_attractor session (2cpr_slowW3 preset) to decode...')
    cfg = SwitchingExperimentConfig(
        seed=0, num_trials=num_trials,
        practice_permutation_repeats=practice_permutation_repeats,
        switch_probs=switch_probs, model_parameters=model_parameters,
    )
    result = run_switching_experiment(cfg)
    trials = result.trials
    n_real = sum(1 for t in trials if not t.is_practice)
    print(f'{len(trials)} trials total, {n_real} real (non-practice) trials.')

    # decode every 5th step to keep runtime reasonable, matching the
    # original notebook's own choice for its live example
    time_steps = np.arange(0, cfg.protocol.number_of_steps, 5)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, population, title in zip(
        axes, ('conjunction', 'feature'),
        ('Model Decoding: Conjunction Units', 'Model Decoding: Feature Units'),
    ):
        print(f'decoding {population} units...')
        relevant, irrelevant = decode_by_relevance(
            trials, population=population, time_steps=time_steps,
        )
        ax.plot(time_steps, relevant * 100, color='crimson', label='relevant')
        ax.plot(time_steps, irrelevant * 100, color='purple', label='irrelevant')
        ax.axhline(50, color='grey', linestyle=':', linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('Time (timesteps)')
        ax.set_ylabel('Classification Accuracy (%)')
        ax.set_ylim(45, 100)
        ax.legend()

    fig.suptitle('gated_attractor (2cpr_slowW3): decoding of relevant vs. irrelevant information')
    fig.tight_layout()

    output_dir = os.path.join(_this_dir, 'gated_attractor', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'decoding_relevant_irrelevant.png')
    fig.savefig(output_path, dpi=150)
    print(f'Saved {output_path}')
