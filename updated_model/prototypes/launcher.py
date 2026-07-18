"""Run the cued task-switching experiment across seeds and plot the results.

OOP counterpart of the flat simulation_launcher.py (which stays in place). It
runs the experiment for each seed, reduces the trials with cued_attractor's
analysis functions, averages across seeds, and writes the same figures.

Run it from the updated_model folder:

    python examples/run_switching.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# make the sibling cued_attractor package importable regardless of the cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cued_attractor import (
    ModelParameters,
    SwitchingExperimentConfig,
    amplifying_eigenvalue_mean_by_kind,
    block_kinds,
    eigenvalue_magnitudes_by_kind,
    incongruence_contrast_by_kind,
    performance_by_block,
    practice_learning_curve,
    run_switching_experiment,
    switch_contrast_by_kind,
)

# output dir for figures (kept separate from the flat launcher's output)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)

# RUN CONSTANTS #
# ===================================================== #

# trials per real block (multiple of 4) and per practice block (multiple of 8)
num_trials = 48
practice_trials = 48

# per-block rule-switch probability, one entry per real block
switch_probs = (0.125, 0.25, 0.5, 0.75) * 2
num_practice_blocks = 2

# number of random seeds (>= ~20 for trustworthy behavioural results)
n_seeds = 20

# MODEL VERSION PARAMETERS #
# ===================================================== #

# named parameter sets for model comparison; add entries to compare against the
# original Whyte model parameters. each value is a ready ModelParameters.
model_versions = {
    'whyte_original': ModelParameters(
        conjunction_lateral_weight=-0.45, conjunction_self_weight=1.0,
        conjunction_to_feature_gain=0.08,
        feature_lateral_weight=-0.28, feature_self_weight=0.73,
        feature_to_conjunction_gain=0.04,
        baseline_activity=0.175,
        fast_learning_rate=0.02, slow_learning_rate=0.0002,
        maximum_fast_weight=1.0, maximum_slow_weight=0.2,
        fast_weight_blend=1.0, slow_weight_blend=1.0,
    ),
}
active_version = 'whyte_original'

# RUN EVERY SEED #
# ===================================================== #

# collect per-seed analysis outputs, then average across seeds below
results = []
for seed in range(n_seeds):
    config = SwitchingExperimentConfig(
        seed=seed,
        num_practice_blocks=num_practice_blocks,
        practice_trials=practice_trials,
        num_trials=num_trials,
        switch_probs=switch_probs,
        model_parameters=model_versions[active_version],
    )
    results.append(run_switching_experiment(config))
    print('Basic task simulation progress', ((seed + 1) / n_seeds) * 100, '%')

kinds = block_kinds(results[0])
kind_labels = [str(kind) for kind in kinds]


def mean_ster(stack):
    # stack has shape (n_seeds, ...); average over seeds (nan-safe for the rare
    # empty trial group), and return mean +/- standard error
    stack = np.asarray(stack, dtype=float)
    mean = np.nanmean(stack, axis=0)
    ster = np.nanstd(stack, axis=0) / np.sqrt(stack.shape[0])
    return mean, ster


def stack_by_kind(per_seed_dicts, attribute=None):
    # turn a list of {kind: value or Contrast} into an array (n_seeds, n_kinds).
    # attribute picks one Contrast field when the values are Contrast objects
    rows = []
    for per_seed in per_seed_dicts:
        if attribute is None:
            rows.append([per_seed[kind] for kind in kinds])
        else:
            rows.append([getattr(per_seed[kind], attribute) for kind in kinds])
    return np.array(rows, dtype=float)


# EIGENVALUE AGGREGATION #
# ===================================================== #

# practice learning curve: (n_seeds, practice_trials x num_practice_blocks)
practice_curves = np.array(
    [practice_learning_curve(result) for result in results], dtype=float
)
mean_prac_sat_vals, ster_prac_sat_vals = mean_ster(practice_curves)

# amplifying-eigenvalue count per block kind
amp_by_kind = [amplifying_eigenvalue_mean_by_kind(r) for r in results]
mean_sat_by_kind, ster_sat_by_kind = mean_ster(stack_by_kind(amp_by_kind))

# pooled eigenvalue magnitudes per block kind (all seeds, trials, units)
eigmag_by_kind = []
for kind in kinds:
    per_kind = [eigenvalue_magnitudes_by_kind(r)[kind] for r in results]
    eigmag_by_kind.append(np.concatenate(per_kind))

# BEHAVIOURAL AGGREGATION #
# ===================================================== #

switch = [switch_contrast_by_kind(r) for r in results]
incong = [incongruence_contrast_by_kind(r) for r in results]

mean_switch_cost_rt, ster_switch_cost_rt = mean_ster(stack_by_kind(switch, 'rt_cost'))
mean_switch_cost_acc, ster_switch_cost_acc = mean_ster(stack_by_kind(switch, 'accuracy_cost'))
mean_incong_cost_rt, ster_incong_cost_rt = mean_ster(stack_by_kind(incong, 'rt_cost'))
mean_incong_cost_acc, ster_incong_cost_acc = mean_ster(stack_by_kind(incong, 'accuracy_cost'))

# pure (non-differenced) RT and accuracy per condition. hard = switch /
# incongruent, easy = repeat / congruent
mean_switch_rt, ster_switch_rt = mean_ster(stack_by_kind(switch, 'hard_rt'))
mean_repeat_rt, ster_repeat_rt = mean_ster(stack_by_kind(switch, 'easy_rt'))
mean_switch_acc, ster_switch_acc = mean_ster(stack_by_kind(switch, 'hard_accuracy'))
mean_repeat_acc, ster_repeat_acc = mean_ster(stack_by_kind(switch, 'easy_accuracy'))
mean_incongruent_rt, ster_incongruent_rt = mean_ster(stack_by_kind(incong, 'hard_rt'))
mean_congruent_rt, ster_congruent_rt = mean_ster(stack_by_kind(incong, 'easy_rt'))
mean_incongruent_acc, ster_incongruent_acc = mean_ster(stack_by_kind(incong, 'hard_accuracy'))
mean_congruent_acc, ster_congruent_acc = mean_ster(stack_by_kind(incong, 'easy_accuracy'))

# PERFORMANCE EVOLUTION #
# ===================================================== #

perf = [performance_by_block(r) for r in results]
block_rt = np.array([rt for rt, _ in perf], dtype=float)
block_acc = np.array([acc for _, acc in perf], dtype=float)
mean_block_rt, ster_block_rt = mean_ster(block_rt)
mean_block_acc, ster_block_acc = mean_ster(block_acc)

# PLOTS #
# ===================================================== #

plt.rcParams['font.size'] = 12

# practice learning curve, with the boundary between the two practice blocks
prac_index = np.arange(0, practice_trials * num_practice_blocks, 1)
fig = plt.figure()
plt.errorbar(prac_index, mean_prac_sat_vals, yerr=ster_prac_sat_vals, color='black', capsize=3, linestyle='-', marker='o')
plt.axvline(practice_trials - .5, color='grey', linestyle='--')
plt.xlabel('Practice trial')
plt.ylabel('Eigenvalues > 1')
plt.xticks(prac_index[::8])
fig.savefig(os.path.join(output_dir, 'practice_eigenvalues_above_1.png'), format='png', dpi=1200)

# all eigenvalue magnitudes per block kind
fig, ax = plt.subplots()
ax.boxplot(eigmag_by_kind, tick_labels=kind_labels, showfliers=False)
ax.set_xlabel('Switch probability (block kind)')
ax.set_ylabel('Eigenvalue magnitude')
fig.savefig(os.path.join(output_dir, 'eigenvalue_magnitudes_by_block_kind.png'), format='png', dpi=1200)

# count of amplifying eigenvalues (>1) per block kind
fig = plt.figure()
plt.errorbar(kinds, mean_sat_by_kind, yerr=ster_sat_by_kind, color='black', capsize=3, linestyle='-', marker='o')
plt.xlabel('Switch probability (block kind)')
plt.ylabel('Amplifying eigenvalues (>1)')
fig.savefig(os.path.join(output_dir, 'amplifying_eigenvalues_by_block_kind.png'), format='png', dpi=1200)

# switch cost (RT and accuracy) per block kind
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].errorbar(kinds, mean_switch_cost_rt, yerr=ster_switch_cost_rt, color='black', capsize=3, linestyle='-', marker='o')
ax[0].set_xlabel('Switch probability (block kind)')
ax[0].set_ylabel('Switch cost: RT (time-steps)')
ax[1].errorbar(kinds, mean_switch_cost_acc, yerr=ster_switch_cost_acc, color='black', capsize=3, linestyle='-', marker='o')
ax[1].set_xlabel('Switch probability (block kind)')
ax[1].set_ylabel('Switch cost: accuracy')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'switch_cost_by_block_kind.png'), format='png', dpi=1200)

# incongruence cost (RT and accuracy) per block kind
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].errorbar(kinds, mean_incong_cost_rt, yerr=ster_incong_cost_rt, color='black', capsize=3, linestyle='-', marker='o')
ax[0].set_xlabel('Switch probability (block kind)')
ax[0].set_ylabel('Incongruence cost: RT (time-steps)')
ax[1].errorbar(kinds, mean_incong_cost_acc, yerr=ster_incong_cost_acc, color='black', capsize=3, linestyle='-', marker='o')
ax[1].set_xlabel('Switch probability (block kind)')
ax[1].set_ylabel('Incongruence cost: accuracy')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'incongruence_cost_by_block_kind.png'), format='png', dpi=1200)

# repeat vs switch RT and accuracy per block kind
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].errorbar(kinds, mean_repeat_rt, yerr=ster_repeat_rt, color='black', capsize=3, linestyle='-', marker='o', label='Repeat')
ax[0].errorbar(kinds, mean_switch_rt, yerr=ster_switch_rt, color='black', capsize=3, linestyle='--', marker='s', label='Switch')
ax[0].set_xlabel('Switch probability (block kind)')
ax[0].set_ylabel('RT (time-steps)')
ax[0].legend()
ax[1].errorbar(kinds, mean_repeat_acc, yerr=ster_repeat_acc, color='black', capsize=3, linestyle='-', marker='o', label='Repeat')
ax[1].errorbar(kinds, mean_switch_acc, yerr=ster_switch_acc, color='black', capsize=3, linestyle='--', marker='s', label='Switch')
ax[1].set_xlabel('Switch probability (block kind)')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'repeat_switch_rt_acc_by_block_kind.png'), format='png', dpi=1200)

# congruent vs incongruent RT and accuracy per block kind
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].errorbar(kinds, mean_congruent_rt, yerr=ster_congruent_rt, color='black', capsize=3, linestyle='-', marker='o', label='Congruent')
ax[0].errorbar(kinds, mean_incongruent_rt, yerr=ster_incongruent_rt, color='black', capsize=3, linestyle='--', marker='s', label='Incongruent')
ax[0].set_xlabel('Switch probability (block kind)')
ax[0].set_ylabel('RT (time-steps)')
ax[0].legend()
ax[1].errorbar(kinds, mean_congruent_acc, yerr=ster_congruent_acc, color='black', capsize=3, linestyle='-', marker='o', label='Congruent')
ax[1].errorbar(kinds, mean_incongruent_acc, yerr=ster_incongruent_acc, color='black', capsize=3, linestyle='--', marker='s', label='Incongruent')
ax[1].set_xlabel('Switch probability (block kind)')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'congruent_incongruent_rt_acc_by_block_kind.png'), format='png', dpi=1200)

# overall RT and accuracy per block index, practice/real boundary marked
num_blocks_total = num_practice_blocks + len(switch_probs)
block_index = np.arange(0, num_blocks_total, 1)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].errorbar(block_index, mean_block_rt, yerr=ster_block_rt, color='black', capsize=3, linestyle='-', marker='o')
ax[0].axvline(num_practice_blocks - .5, color='grey', linestyle='--')
ax[0].set_xlabel('Block')
ax[0].set_ylabel('RT (time-steps)')
ax[0].set_xticks(block_index)
ax[1].errorbar(block_index, mean_block_acc, yerr=ster_block_acc, color='black', capsize=3, linestyle='-', marker='o')
ax[1].axvline(num_practice_blocks - .5, color='grey', linestyle='--')
ax[1].set_xlabel('Block')
ax[1].set_ylabel('Accuracy')
ax[1].set_xticks(block_index)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'performance_evolution_by_block.png'), format='png', dpi=1200)

plt.show()
