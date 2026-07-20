"""Benchmark: cued_attractor vs. gated_attractor, both run on their own
native cued task-switching paradigm.

Two conditions, each using its own package's active model-version
parameters (see each package's model_versions_config.py) and an identical
switch-probability schedule that includes a switch_probability == 0.0
baseline:

  cued_attractor  - cues signal the rule; no gating population. Active
                     version: CUED_VERSION.
  gated_attractor - the same cued paradigm, plus a gating population that
                     learns, from the cue, to suppress the task-irrelevant
                     colour/shape pair. Active version: GATED_VERSION.

Both packages share the same experiment.py paradigm (instruction blocks,
performance-practice blocks, then real cued-switching blocks) and the same
analysis.py function names, so nothing here reimplements contrast or
accuracy logic -- every number comes straight from each package's own
summarize_behavior / switch_contrast_by_kind / incongruence_contrast_by_kind
/ amplifying_eigenvalue_mean_by_kind / block_kinds.

This replaces an earlier draft that instead compared gated_attractor
against the original published Whyte et al. model on an artificial no-cue
switching schedule grafted onto that (block-only) architecture -- comparing
cued_attractor vs. gated_attractor instead isolates the actual question
(does the gating mechanism help) on a paradigm both models run natively, no
grafting required.

Run from the updated_model folder:

    python3 benchmark_cued_vs_gated.py --seeds 20
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

import numpy as np

# both packages are siblings of this file
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from cued_attractor import (  # noqa: E402
    SwitchingExperimentConfig as CuedConfig,
    amplifying_eigenvalue_mean_by_kind as cued_amp_eig_by_kind,
    block_kinds as cued_block_kinds,
    incongruence_contrast_by_kind as cued_incong_by_kind,
    run_switching_experiment as run_cued_experiment,
    summarize_behavior as cued_summarize_behavior,
    switch_contrast_by_kind as cued_switch_by_kind,
)
from cued_attractor.model_versions_config import model_versions as cued_model_versions  # noqa: E402

from gated_attractor import (  # noqa: E402
    SwitchingExperimentConfig as GatedConfig,
    amplifying_eigenvalue_mean_by_kind as gated_amp_eig_by_kind,
    block_kinds as gated_block_kinds,
    incongruence_contrast_by_kind as gated_incong_by_kind,
    run_switching_experiment as run_gated_experiment,
    summarize_behavior as gated_summarize_behavior,
    switch_contrast_by_kind as gated_switch_by_kind,
)
from gated_attractor.model_versions_config import model_versions as gated_model_versions  # noqa: E402

# RUN CONSTANTS #
# ===================================================== #

CUED_VERSION = 'whyte_params_2cpr'  # matches cued_attractor/launcher.py's active_versions
GATED_VERSION = '2cpr_slowW3'  # matches gated_attractor/launcher.py's active_versions

# shared across both conditions so their per-switch-probability breakdowns
# line up point for point. 0.0 up front is the no-switching baseline.
SWITCH_PROBS = (0.0, 0.125, 0.25, 0.5, 0.75) * 2


def run_condition_cued(seed: int):
    config = CuedConfig(seed=seed, model_parameters=cued_model_versions[CUED_VERSION], switch_probs=SWITCH_PROBS,
                        practice_permutation_repeats=5)
    return run_cued_experiment(config)


def run_condition_gated(seed: int):
    config = GatedConfig(seed=seed, model_parameters=gated_model_versions[GATED_VERSION], switch_probs=SWITCH_PROBS,
                         practice_permutation_repeats=5)
    return run_gated_experiment(config)


# PER-SEED SUMMARIES #
# ===================================================== #

def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float('nan')


def _mean_sem(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)), float(np.nanstd(array) / np.sqrt(len(array)))


def _pooled_from_by_kind(by_kind: dict) -> tuple[float, float]:
    """Equal-weighted mean of rt_cost / accuracy_cost across every switch-
    probability kind. The switch_probability == 0.0 kind has no rule-switch
    trials to contrast, so its Contrast is nan there; nanmean skips it."""

    rt_costs = [contrast.rt_cost for contrast in by_kind.values()]
    accuracy_costs = [contrast.accuracy_cost for contrast in by_kind.values()]
    return float(np.nanmean(rt_costs)), float(np.nanmean(accuracy_costs))


@dataclass
class ConditionSummary:
    accuracy: float
    switch_rt_cost: float
    switch_acc_cost: float
    incongruence_rt_cost: float
    incongruence_acc_cost: float
    # both packages record one amplifying-eigenvalue count per block, and
    # block 0 / block 1 are the two instruction blocks in both -- the one
    # directly matched formation-speed comparison between them.
    eigenvalues_after_first_rule: int
    eigenvalues_after_both_rules: int
    final_amplifying_eigenvalues: int


def summarize_cued_result(result) -> ConditionSummary:
    real_trials = [t for t in result.trials if not t.is_practice]
    behavior = cued_summarize_behavior(real_trials)
    switch_rt, switch_acc = _pooled_from_by_kind(cued_switch_by_kind(result))
    incong_rt, incong_acc = _pooled_from_by_kind(cued_incong_by_kind(result))
    return ConditionSummary(
        accuracy=behavior.accuracy,
        switch_rt_cost=switch_rt,
        switch_acc_cost=switch_acc,
        incongruence_rt_cost=incong_rt,
        incongruence_acc_cost=incong_acc,
        eigenvalues_after_first_rule=result.amplifying_eigenvalue_count_by_block[0],
        eigenvalues_after_both_rules=result.amplifying_eigenvalue_count_by_block[1],
        final_amplifying_eigenvalues=result.amplifying_eigenvalue_count_by_block[-1],
    )


def summarize_gated_result(result) -> ConditionSummary:
    real_trials = [t for t in result.trials if not t.is_practice]
    behavior = gated_summarize_behavior(real_trials)
    switch_rt, switch_acc = _pooled_from_by_kind(gated_switch_by_kind(result))
    incong_rt, incong_acc = _pooled_from_by_kind(gated_incong_by_kind(result))
    return ConditionSummary(
        accuracy=behavior.accuracy,
        switch_rt_cost=switch_rt,
        switch_acc_cost=switch_acc,
        incongruence_rt_cost=incong_rt,
        incongruence_acc_cost=incong_acc,
        eigenvalues_after_first_rule=result.amplifying_eigenvalue_count_by_block[0],
        eigenvalues_after_both_rules=result.amplifying_eigenvalue_count_by_block[1],
        final_amplifying_eigenvalues=result.amplifying_eigenvalue_count_by_block[-1],
    )


def run_all(n_seeds: int):
    print(f'Running cued_attractor ({CUED_VERSION}, {n_seeds} seeds, switch_probs={SWITCH_PROBS})...')
    cued_results = [run_condition_cued(seed) for seed in range(n_seeds)]

    print(f'Running gated_attractor ({GATED_VERSION}, {n_seeds} seeds, switch_probs={SWITCH_PROBS})...')
    gated_results = [run_condition_gated(seed) for seed in range(n_seeds)]

    summaries = {
        'cued_attractor': [summarize_cued_result(r) for r in cued_results],
        'gated_attractor': [summarize_gated_result(r) for r in gated_results],
    }
    eigenvalue_trajectories = {
        'cued_attractor': [np.array(r.amplifying_eigenvalue_count_by_block) for r in cued_results],
        'gated_attractor': [np.array(r.amplifying_eigenvalue_count_by_block) for r in gated_results],
    }
    return summaries, eigenvalue_trajectories, cued_results, gated_results


# REPORTING #
# ===================================================== #

_TABLE_FIELDS = [
    ('accuracy', 'Accuracy'),
    ('switch_rt_cost', 'Switch cost: RT (steps)'),
    ('switch_acc_cost', 'Switch cost: accuracy'),
    ('incongruence_rt_cost', 'Incongruence cost: RT (steps)'),
    ('incongruence_acc_cost', 'Incongruence cost: accuracy'),
    ('eigenvalues_after_first_rule', 'Amplifying eigenvalues (after rule 1 taught)'),
    ('eigenvalues_after_both_rules', 'Amplifying eigenvalues (after both rules taught)'),
    ('final_amplifying_eigenvalues', 'Amplifying eigenvalues (final block)'),
]


def print_summary_table(summaries: dict[str, list[ConditionSummary]]) -> None:
    names = list(summaries)
    print()
    print(f"{'Metric':<38s}" + ''.join(f'{name:>22s}' for name in names))
    for attr, label in _TABLE_FIELDS:
        row = f'{label:<38s}'
        for name in names:
            values = [getattr(s, attr) for s in summaries[name]]
            mean, sem = _mean_sem(values)
            row += f'{mean:>13.3f} +/-{sem:<6.3f}'
        print(row)


def print_breakdown_by_kind(
    condition_name, results, switch_by_kind_fn, incong_by_kind_fn, amp_eig_by_kind_fn, kinds_fn,
) -> None:
    kinds = kinds_fn(results[0])
    print(f'\n{condition_name}: switch/incongruence cost and amplifying eigenvalues by switch probability')
    print(f"{'p(switch)':>10s}{'switch RT':>14s}{'switch acc':>14s}"
          f"{'incong RT':>14s}{'incong acc':>14s}{'amp. eig.':>12s}")
    for kind in kinds:
        switch_costs = [switch_by_kind_fn(r)[kind] for r in results]
        incong_costs = [incong_by_kind_fn(r)[kind] for r in results]
        eig_values = [amp_eig_by_kind_fn(r)[kind] for r in results]

        switch_rt_mean, _ = _mean_sem([c.rt_cost for c in switch_costs])
        switch_acc_mean, _ = _mean_sem([c.accuracy_cost for c in switch_costs])
        incong_rt_mean, _ = _mean_sem([c.rt_cost for c in incong_costs])
        incong_acc_mean, _ = _mean_sem([c.accuracy_cost for c in incong_costs])
        eig_mean, _ = _mean_sem(eig_values)

        print(f'{kind:>10.3f}{switch_rt_mean:>14.2f}{switch_acc_mean:>14.3f}'
              f'{incong_rt_mean:>14.2f}{incong_acc_mean:>14.3f}{eig_mean:>12.2f}')


def save_csv(summaries: dict[str, list[ConditionSummary]], path: str) -> None:
    fieldnames = ['condition', 'seed'] + [attr for attr, _ in _TABLE_FIELDS]
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        for name, condition_summaries in summaries.items():
            for seed, summary in enumerate(condition_summaries):
                writer.writerow([name, seed] + [getattr(summary, attr) for attr, _ in _TABLE_FIELDS])
    print(f'\nPer-seed data written to {path}')


# ACCURACY BY SWITCH PROBABILITY #
# ===================================================== #

def accuracy_by_kind(results, kinds_fn) -> dict[float, list[float]]:
    """Per-seed accuracy at each switch probability. Computed directly from
    TrialResult.switch_probability / .correct, which both packages provide
    identically, rather than a package-specific helper."""

    kinds = kinds_fn(results[0])
    accuracy: dict[float, list[float]] = {kind: [] for kind in kinds}
    for result in results:
        for kind in kinds:
            trials = [t for t in result.trials if not t.is_practice and t.switch_probability == kind]
            accuracy[kind].append(_mean_or_nan([float(t.correct) for t in trials]))
    return accuracy


def save_accuracy_by_kind_csv(cued_results, gated_results, path: str) -> None:
    cued_acc = accuracy_by_kind(cued_results, cued_block_kinds)
    gated_acc = accuracy_by_kind(gated_results, gated_block_kinds)
    kinds = sorted(cued_acc)
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['condition', 'seed', 'switch_probability', 'accuracy'])
        for kind in kinds:
            for seed, value in enumerate(cued_acc[kind]):
                writer.writerow(['cued_attractor', seed, kind, value])
            for seed, value in enumerate(gated_acc[kind]):
                writer.writerow(['gated_attractor', seed, kind, value])
    print(f'Accuracy-by-switch-probability data written to {path}')


def plot_accuracy_by_switch_probability(cued_results, gated_results, path: str) -> None:
    """Accuracy vs. switch probability, cued_attractor vs. gated_attractor,
    mean +/- s.e.m. across seeds. The headline figure for whether gating
    helps, and how the benefit (if any) scales with switch probability."""

    import matplotlib.pyplot as plt

    plt.rcParams['font.size'] = 12
    cued_acc = accuracy_by_kind(cued_results, cued_block_kinds)
    gated_acc = accuracy_by_kind(gated_results, gated_block_kinds)
    kinds = sorted(cued_acc)

    cued_mean, cued_sem = zip(*(_mean_sem(cued_acc[k]) for k in kinds))
    gated_mean, gated_sem = zip(*(_mean_sem(gated_acc[k]) for k in kinds))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(kinds, cued_mean, yerr=cued_sem, color='tab:orange', capsize=3,
                linestyle='-', marker='o', label='cued_attractor (no gating)')
    ax.errorbar(kinds, gated_mean, yerr=gated_sem, color='tab:blue', capsize=3,
                linestyle='-', marker='o', label='gated_attractor (gating)')
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=1, label='chance')
    ax.set_xlabel('Switch probability')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'Accuracy-by-switch-probability figure written to {path}')


# EIGENVALUE-TRAJECTORY (ATTRACTOR-FORMATION SPEED) PLOT #
# ===================================================== #

def plot_eigenvalue_trajectories(eigenvalue_trajectories: dict[str, list[np.ndarray]], path: str) -> None:
    """Two panels: (1) the structurally-matched after-rule-1 / after-both-
    rules comparison (the numbers also in the summary table), as bars; (2)
    each condition's full per-block trajectory (mean +/- s.e.m.). Both
    conditions have the same block structure here (2 instruction + 2
    performance-practice + len(SWITCH_PROBS) real blocks), so this is a
    directly matched trajectory, unlike the block-index caveat that applied
    when the published Whyte model was one of the conditions.
    """

    import matplotlib.pyplot as plt

    plt.rcParams['font.size'] = 12
    names = list(eigenvalue_trajectories)
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(11, 4.5))

    bar_width = 0.8 / len(names)
    x_positions = np.arange(2)
    for i, name in enumerate(names):
        stacked = np.stack([traj[:2] for traj in eigenvalue_trajectories[name]])
        mean, sem = stacked.mean(axis=0), stacked.std(axis=0) / np.sqrt(stacked.shape[0])
        offset = (i - (len(names) - 1) / 2) * bar_width
        ax_bar.bar(x_positions + offset, mean, bar_width, yerr=sem, capsize=3,
                   color=colors[i], label=name)
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(['After rule 1 taught', 'After both rules taught'])
    ax_bar.set_ylabel('Amplifying eigenvalues (> 1)')
    ax_bar.legend(fontsize=9)

    for i, name in enumerate(names):
        max_len = max(len(traj) for traj in eigenvalue_trajectories[name])
        padded = np.full((len(eigenvalue_trajectories[name]), max_len), np.nan)
        for row, traj in enumerate(eigenvalue_trajectories[name]):
            padded[row, :len(traj)] = traj
        mean = np.nanmean(padded, axis=0)
        sem = np.nanstd(padded, axis=0) / np.sqrt(np.sum(~np.isnan(padded), axis=0))
        block_index = np.arange(max_len)
        ax_line.plot(block_index, mean, color=colors[i], marker='o', markersize=3, label=name)
        ax_line.fill_between(block_index, mean - sem, mean + sem, color=colors[i], alpha=0.2)
    ax_line.axvline(1.5, color='grey', linestyle='--', linewidth=1)
    ax_line.set_xlabel('Block index')
    ax_line.set_ylabel('Amplifying eigenvalues (> 1)')
    ax_line.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f'Eigenvalue-trajectory figure written to {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=20, help='number of random seeds per condition')
    parser.add_argument('--output', type=str, default=os.path.join(_HERE, 'benchmark_output.csv'),
                         help='where to write the per-seed CSV')
    parser.add_argument('--eigenvalue-plot', type=str,
                         default=os.path.join(_HERE, 'benchmark_eigenvalue_trajectories.png'),
                         help='where to write the eigenvalue-trajectory figure')
    parser.add_argument('--accuracy-output', type=str,
                         default=os.path.join(_HERE, 'benchmark_accuracy_by_switch_probability.csv'),
                         help='where to write the per-seed accuracy-by-switch-probability CSV')
    parser.add_argument('--accuracy-plot', type=str,
                         default=os.path.join(_HERE, 'benchmark_accuracy_by_switch_probability.png'),
                         help='where to write the accuracy-by-switch-probability figure')
    args = parser.parse_args()

    all_summaries, all_eigenvalue_trajectories, cued_raw_results, gated_raw_results = run_all(args.seeds)

    print_summary_table(all_summaries)
    print_breakdown_by_kind('cued_attractor', cued_raw_results, cued_switch_by_kind, cued_incong_by_kind,
                             cued_amp_eig_by_kind, cued_block_kinds)
    print_breakdown_by_kind('gated_attractor', gated_raw_results, gated_switch_by_kind, gated_incong_by_kind,
                             gated_amp_eig_by_kind, gated_block_kinds)
    save_csv(all_summaries, args.output)
    plot_eigenvalue_trajectories(all_eigenvalue_trajectories, args.eigenvalue_plot)

    save_accuracy_by_kind_csv(cued_raw_results, gated_raw_results, args.accuracy_output)
    plot_accuracy_by_switch_probability(cued_raw_results, gated_raw_results, args.accuracy_plot)
