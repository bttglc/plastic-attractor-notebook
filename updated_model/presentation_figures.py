"""Presentation figures for the gated-attractor TMS demo.

Run this from the ``updated_model`` directory:

    python presentation_figures.py

The script builds a warmed-up gated-attractor model, runs one Control trial and
one conjunction-only TMS trial, and then renders two presentation-ready figures:

1. a side-by-side comparison of conjunction-unit trajectories over time;
2. a 2D PCA phase portrait of the perturbed trial that highlights attractor
   hijacking.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gated_attractor import (  # noqa: E402
    EpochProtocol,
    Feature,
    ModelParameters,
    PlasticAttractor,
    Task,
    TMSPerturbation,
    build_vocabulary,
)
from gated_attractor.experiment import (  # noqa: E402
    PRACTICE_TASKS,
    SwitchingExperimentConfig,
    TransitionType,
    _run_block,
    _run_trial,
    practice_block_plan,
    real_block_plan,
)


# Presentation styling: larger type, cleaner lines, and bold titles.
plt.rcParams.update(
    {
        'font.size': 14,
        'axes.titlesize': 20,
        'axes.titleweight': 'bold',
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
        'figure.titlesize': 22,
    }
)


SEED = 0
PRACTICE_PERMUTATION_REPEATS = 6
DEMO_TRIAL_COUNT = 4

# These parameters match the validated gated-attractor configuration used for
# the TMS demo: gating is on, and the gating gains are the swept values that
# were already checked against the learned-rule behaviour.
MODEL_PARAMETERS = ModelParameters(
    number_of_gating_units=2,
    gating_to_feature_gain=0.7,
    gating_to_relevant_feature_gain=0.6,
)
PROTOCOL = EpochProtocol()

# A light, presentation-friendly palette for the unit trajectories.
CONJUNCTION_COLORS = [
    '#1b9e77',
    '#d95f02',
    '#7570b3',
    '#e7298a',
]


def _build_warmed_up_model() -> tuple[PlasticAttractor, SwitchingExperimentConfig, object, object]:
    """Warm up a model with the standard practice blocks.

    The returned model has already completed the instruction and performance-
    practice phases. The demo trials later in the script use a deep copy of
    this warmed-up state so the control and perturbed conditions start from the
    same learned weights and activity state.
    """

    random_generator = np.random.RandomState(SEED)
    config = SwitchingExperimentConfig(
        seed=SEED,
        num_practice_blocks=2,
        practice_permutation_repeats=PRACTICE_PERMUTATION_REPEATS,
        num_trials=DEMO_TRIAL_COUNT,
        switch_probs=(0.0,),
        model_parameters=MODEL_PARAMETERS,
        protocol=PROTOCOL,
        perturbation=None,
        learn_during_trials=True,
    )
    vocabulary = build_vocabulary(MODEL_PARAMETERS.num_cues_per_rule)
    model = PlasticAttractor(parameters=MODEL_PARAMETERS, random_generator=random_generator)

    for block_index, task in enumerate(PRACTICE_TASKS):
        plan = practice_block_plan(
            task,
            config.practice_permutation_repeats,
            vocabulary,
            random_generator,
        )
        _run_block(
            model,
            config,
            vocabulary,
            block_index=block_index,
            is_practice=True,
            apply_teaching=True,
            switch_probability=None,
            plan=plan,
        )

    for practice_index, task in enumerate(PRACTICE_TASKS):
        block_index = config.num_practice_blocks + practice_index
        plan = real_block_plan(
            switch_probability=0.0,
            num_trials=config.num_trials,
            vocabulary=vocabulary,
            random_generator=random_generator,
            initial_task=task,
        )
        _run_block(
            model,
            config,
            vocabulary,
            block_index=block_index,
            is_practice=True,
            apply_teaching=False,
            switch_probability=None,
            plan=plan,
        )

    demo_plan = real_block_plan(
        switch_probability=0.0,
        num_trials=config.num_trials,
        vocabulary=vocabulary,
        random_generator=random_generator,
        initial_task=Task.COLOR,
    )

    return model, config, vocabulary, demo_plan[0]


def _build_perturbation(protocol: EpochProtocol) -> TMSPerturbation:
    """Construct the conjunction-only TMS condition used in the comparison."""

    # Keep the pulse squarely inside the stimulus window so the effect is easy
    # to see in the trajectories.
    start = protocol.stimulus_window.start + 10
    stop = protocol.stimulus_window.start + 25
    return TMSPerturbation.conjunction_only(start, stop, clamp_value=0.0)


def _run_condition_trial(
    *,
    model: PlasticAttractor,
    config: SwitchingExperimentConfig,
    vocabulary,
    planned_trial,
    perturbation: TMSPerturbation | None,
):
    """Run one demo trial under a chosen perturbation.

    Learning is disabled for the presentation trial so the comparison reflects
    the same warmed-up network state at the start of each condition.
    """

    demo_config = replace(
        config,
        perturbation=perturbation,
        learn_during_trials=False,
    )

    trial = _run_trial(
        model=model,
        config=demo_config,
        vocabulary=vocabulary,
        block_index=999,
        trial_index=0,
        is_practice=False,
        apply_teaching=False,
        switch_probability=0.0,
        planned=planned_trial,
        transition_type=TransitionType.FIRST_TRIAL,
    )
    return trial


def _condition_trials():
    """Return the warmed-up control and conjunction-only TMS trials.

    The model is deep-copied so both conditions start from exactly the same
    learned weights, activity state, and RNG state.
    """

    model, config, vocabulary, planned_trial = _build_warmed_up_model()
    perturbation = _build_perturbation(config.protocol)

    control_trial = _run_condition_trial(
        model=copy.deepcopy(model),
        config=config,
        vocabulary=vocabulary,
        planned_trial=planned_trial,
        perturbation=None,
    )
    tms_trial = _run_condition_trial(
        model=copy.deepcopy(model),
        config=config,
        vocabulary=vocabulary,
        planned_trial=planned_trial,
        perturbation=perturbation,
    )
    return control_trial, tms_trial, perturbation


def _add_time_window_shading(ax, perturbation: TMSPerturbation) -> None:
    ax.axvspan(
        perturbation.window.start,
        perturbation.window.stop,
        color='#f4a3a3',
        alpha=0.28,
        label='TMS window',
        zorder=0,
    )


def plot_conjunction_trajectories(control_trial, tms_trial, perturbation: TMSPerturbation):
    """Plot the four conjunction-unit trajectories side by side."""

    time = np.arange(control_trial.trajectory.conjunction_activity.shape[0])
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 5.8),
        sharey=True,
        constrained_layout=True,
    )

    panels = [
        ('Control trial', control_trial, None),
        ('Perturbed trial: conjunction-only TMS', tms_trial, perturbation),
    ]

    for ax, (title, trial, maybe_perturbation) in zip(axes, panels):
        for unit_index in range(trial.trajectory.conjunction_activity.shape[1]):
            ax.plot(
                time,
                trial.trajectory.conjunction_activity[:, unit_index],
                color=CONJUNCTION_COLORS[unit_index],
                linewidth=2.4,
                label=f'Conjunction {unit_index + 1}',
            )

        if maybe_perturbation is not None:
            _add_time_window_shading(ax, maybe_perturbation)

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.15, linewidth=0.8)
        ax.legend(frameon=False, loc='upper right', ncol=2)

    axes[0].set_ylabel('Conjunction activity')
    fig.suptitle('Control vs TMS perturbation', fontweight='bold')
    return fig


def _pca_2d(conjunction_activity: np.ndarray) -> np.ndarray:
    """Project the 4D conjunction trajectory into 2D using PCA via SVD."""

    centered = conjunction_activity - conjunction_activity.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


def _time_colored_phase_portrait(ax, coordinates: np.ndarray):
    """Draw a phase portrait with a time gradient and return the line collection."""

    points = coordinates.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    time_gradient = LinearSegmentedColormap.from_list(
        'presentation_time_blue',
        ['#b7d9f2', '#08306b'],
    )
    line_collection = LineCollection(
        segments,
        cmap=time_gradient,
        norm=Normalize(0, len(coordinates) - 2),
        linewidth=3.0,
        alpha=0.95,
    )
    line_collection.set_array(np.arange(len(coordinates) - 1))
    ax.add_collection(line_collection)
    return line_collection


def plot_attractor_hijacking(tms_trial, perturbation: TMSPerturbation):
    """Visualise how the perturbed 4D conjunction trajectory moves in PCA space."""

    coordinates = _pca_2d(tms_trial.trajectory.conjunction_activity)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)

    line_collection = _time_colored_phase_portrait(ax, coordinates)
    ax.autoscale()
    ax.set_aspect('equal', adjustable='datalim')

    # Mark the start and end of the pulse so the audience can see the state
    # being knocked into a new basin and then evolving after the clamp ends.
    start_index = perturbation.window.start
    end_index = perturbation.window.stop
    ax.scatter(
        coordinates[start_index, 0],
        coordinates[start_index, 1],
        s=140,
        marker='o',
        facecolor='white',
        edgecolor='#cc0000',
        linewidth=2.2,
        zorder=5,
        label='TMS onset',
    )
    ax.scatter(
        coordinates[end_index, 0],
        coordinates[end_index, 1],
        s=160,
        marker='s',
        facecolor='#cc0000',
        edgecolor='black',
        linewidth=1.4,
        zorder=5,
        label='TMS offset',
    )

    ax.set_title('Attractor hijacking in PCA space', fontweight='bold')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.grid(alpha=0.14, linewidth=0.8)
    ax.legend(frameon=False, loc='best')
    fig.colorbar(line_collection, ax=ax, pad=0.02, label='Timestep')
    return fig


def main() -> None:
    control_trial, tms_trial, perturbation = _condition_trials()

    print(f"Control recovery_latency = {control_trial.recovery_latency_in_steps}")
    print(f"Perturbed recovery_latency = {tms_trial.recovery_latency_in_steps}")

    plot_conjunction_trajectories(control_trial, tms_trial, perturbation)
    plot_attractor_hijacking(tms_trial, perturbation)
    plt.show()


if __name__ == '__main__':
    main()
