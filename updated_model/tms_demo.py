"""Small demo for the gated-attractor TMS perturbation conditions.

Run this from the ``updated_model`` directory:

    python tms_demo.py

The script does three things for each condition:
1. warms up a fresh model with the usual practice blocks,
2. runs one demo trial with a chosen TMS perturbation,
3. prints the trial's recovery latency and plots conjunction/gating activity.
"""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from gated_attractor import (
    EpochProtocol,
    ModelParameters,
    PlasticAttractor,
    Task,
    TMSPerturbation,
    build_vocabulary,
)
from gated_attractor.experiment import (
    PRACTICE_TASKS,
    SwitchingExperimentConfig,
    TransitionType,
    _run_block,
    _run_trial,
    practice_block_plan,
    real_block_plan,
)


# Keep the demo close to the validated gated-attractor settings.
PARAMETERS = ModelParameters(
    number_of_gating_units=2,
    gating_to_feature_gain=0.7,
    gating_to_relevant_feature_gain=0.6,
)
PROTOCOL = EpochProtocol()
SEED = 0
PRACTICE_PERMUTATION_REPEATS = 6
DEMO_TRIAL_COUNT = 4


def _build_warmed_up_model() -> tuple[PlasticAttractor, SwitchingExperimentConfig, object, object]:
    """Train a fresh model, then return it together with a demo trial plan.

    The returned model has already completed the standard practice blocks, so
    the single demo trial can focus on the TMS perturbation rather than the
    learning warm-up.
    """

    random_generator = np.random.RandomState(SEED)
    config = SwitchingExperimentConfig(
        seed=SEED,
        num_practice_blocks=2,
        practice_permutation_repeats=PRACTICE_PERMUTATION_REPEATS,
        num_trials=DEMO_TRIAL_COUNT,
        switch_probs=(0.0,),
        model_parameters=PARAMETERS,
        protocol=PROTOCOL,
        perturbation=None,
    )
    vocabulary = build_vocabulary(PARAMETERS.num_cues_per_rule)
    model = PlasticAttractor(parameters=PARAMETERS, random_generator=random_generator)

    # Two instruction blocks: one teaches color, the other teaches shape.
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

    # Two performance-practice blocks keep learning on but remove the explicit
    # teaching drive, matching the normal experiment structure.
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

    # One deterministic demo trial is enough to compare the TMS conditions.
    demo_plan = real_block_plan(
        switch_probability=0.0,
        num_trials=config.num_trials,
        vocabulary=vocabulary,
        random_generator=random_generator,
        initial_task=Task.COLOR,
    )

    return model, config, vocabulary, demo_plan[0]


def _make_perturbation(condition: str, protocol: EpochProtocol) -> TMSPerturbation:
    """Create the requested TMS condition using one shared window."""

    # This window sits inside the stimulus period so the clamp is easy to see.
    start = protocol.stimulus_window.start + 10
    stop = protocol.stimulus_window.start + 25

    if condition == "conjunction_only":
        return TMSPerturbation.conjunction_only(start, stop, clamp_value=0.0)
    if condition == "gating_only":
        return TMSPerturbation.gating_only(start, stop, clamp_value=0.0)
    if condition == "both":
        return TMSPerturbation.both(start, stop, clamp_value=0.0)

    raise ValueError(f"Unknown condition: {condition!r}")


def _run_demo_trial(condition: str):
    """Run one warmed-up trial under the requested perturbation."""

    model, base_config, vocabulary, planned_trial = _build_warmed_up_model()
    perturbation = _make_perturbation(condition, base_config.protocol)
    demo_config = replace(base_config, perturbation=perturbation)

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
    return trial, perturbation


def _plot_trial(condition: str, trial, perturbation: TMSPerturbation) -> None:
    """Plot conjunction and gating activity over the full trial timeline."""

    steps = np.arange(trial.trajectory.conjunction_activity.shape[0])
    window = perturbation.window

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 7),
        constrained_layout=True,
    )

    for unit_index in range(trial.trajectory.conjunction_activity.shape[1]):
        axes[0].plot(
            steps,
            trial.trajectory.conjunction_activity[:, unit_index],
            linewidth=1.8,
            drawstyle="steps-post",
            label=f"conjunction {unit_index + 1}",
        )

    for gate_index in range(trial.trajectory.gating_activity.shape[1]):
        axes[1].plot(
            steps,
            trial.trajectory.gating_activity[:, gate_index],
            linewidth=1.8,
            drawstyle="steps-post",
            label=f"gate {gate_index + 1}",
        )

    for ax, title in (
        (axes[0], "Conjunction activity"),
        (axes[1], "Gating activity"),
    ):
        ax.axvspan(window.start, window.stop, color="tab:red", alpha=0.14, label="TMS window")
        ax.axhline(0.0, color="0.35", linewidth=1.0, linestyle=":")
        ax.set_ylabel("activity")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(frameon=False, ncol=2, loc="upper right")

    axes[1].set_xlabel("timestep")
    fig.suptitle(f"TMS condition: {condition} | recovery_latency_in_steps = {trial.recovery_latency_in_steps}")
    plt.show()


def main() -> None:
    conditions = ["conjunction_only", "gating_only", "both"]

    for condition in conditions:
        trial, perturbation = _run_demo_trial(condition)
        print(f"{condition}: recovery_latency_in_steps = {trial.recovery_latency_in_steps}")
        _plot_trial(condition, trial, perturbation)


if __name__ == "__main__":
    main()
