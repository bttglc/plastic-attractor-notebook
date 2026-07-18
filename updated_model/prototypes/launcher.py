"""Run the cued task-switching experiment across seeds and plot the results.

OOP counterpart of the flat simulation_launcher.py (which stays in place). It
runs the experiment for each seed, reduces the trials with cued_attractor's
analysis functions, averages across seeds, and writes the same figures.

Run it from the updated_model folder:

    python prototypes/launcher.py
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

# make the sibling cued_attractor package importable regardless of the cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cued_attractor import (
    PRACTICE_TASKS,
    Feature,
    SwitchingExperimentConfig,
    Task,
    amplifying_eigenvalue_mean_by_kind,
    block_kinds,
    build_vocabulary,
    eigenvalue_magnitudes_by_kind,
    incongruence_contrast_by_kind,
    performance_by_block,
    practice_learning_curve,
    run_switching_experiment,
    switch_contrast_by_kind,
)
from model_versions_config import model_versions

# output dir for figures (kept separate from the flat launcher's output);
# subdir named after each active model version so different versions don't
# overwrite each other's figures
base_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# RUN CONSTANTS #
# ===================================================== #

# trials per real block (multiple of 4); practice runs the full (cue x
# stimulus) permutation practice_permutation_repeats times per block
num_trials = 48
practice_permutation_repeats = 1

# per-block rule-switch probability, one entry per real block
switch_probs = (0.125, 0.25, 0.5, 0.75) * 2
num_practice_blocks = 2
# probability of a mid-block rule switch during practice; 0.0 = practice
# blocks each teach one fixed rule and never switch (the published design)
practice_switch_probability = 0.0

# number of random seeds (>= ~20 for trustworthy behavioural results)
n_seeds = 20

# MODEL VERSIONS #
# ===================================================== #

# named parameter sets live in model_versions_config.py; pick any subset of
# model_versions.keys() to run in this invocation
active_versions = ['whyte_params_2cpr', '2cpr_Wslow_cap_high']

# PARALLEL SEED EXECUTION #
# ===================================================== #

# number of available CPUs; if >= 20 we also parallelise across model
# versions (one big pool of (version, seed) jobs), otherwise versions run
# sequentially and only their seeds are parallelised
cpu_count = os.cpu_count() or 1


def build_configs(params):
    # one SwitchingExperimentConfig per seed, sharing every run constant
    # above except the model parameters and the seed itself
    return [
        SwitchingExperimentConfig(
            seed=seed,
            num_practice_blocks=num_practice_blocks,
            practice_permutation_repeats=practice_permutation_repeats,
            practice_switch_probability=practice_switch_probability,
            num_trials=num_trials,
            switch_probs=switch_probs,
            model_parameters=params,
        )
        for seed in range(n_seeds)
    ]


def run_seeds_parallel(configs, max_workers):
    # executor.map preserves input order, so results[i] still corresponds to
    # seed=i (needed downstream, e.g. snapshot_seed = 0)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(run_switching_experiment, configs))


def mean_ster(stack):
    # stack has shape (n_seeds, ...); average over seeds (nan-safe for the rare
    # empty trial group), and return mean +/- standard error
    stack = np.asarray(stack, dtype=float)
    mean = np.nanmean(stack, axis=0)
    ster = np.nanstd(stack, axis=0) / np.sqrt(stack.shape[0])
    return mean, ster


def stack_by_kind(per_seed_dicts, kinds, attribute=None):
    # turn a list of {kind: value or Contrast} into an array (n_seeds, n_kinds).
    # attribute picks one Contrast field when the values are Contrast objects
    rows = []
    for per_seed in per_seed_dicts:
        if attribute is None:
            rows.append([per_seed[kind] for kind in kinds])
        else:
            rows.append([getattr(per_seed[kind], attribute) for kind in kinds])
    return np.array(rows, dtype=float)


def analyze_and_plot(version_name, results):
    # aggregate one model version's per-seed results and write its figures
    # to base_output_dir/<version_name>/
    output_dir = os.path.join(base_output_dir, version_name)
    os.makedirs(output_dir, exist_ok=True)

    # read the unit counts off the actual run config rather than the shared
    # model_versions_config defaults, so plotting stays correct even if a
    # version entry overrides either value
    params = results[0].config.model_parameters
    num_cues_per_rule = params.num_cues_per_rule
    number_of_conjunction_units = params.number_of_conjunction_units

    kinds = block_kinds(results[0])
    kind_labels = [str(kind) for kind in kinds]

    # EIGENVALUE AGGREGATION #
    # ===================================================== #

    # practice learning curve: (n_seeds, practice_trials x num_practice_blocks)
    practice_curves = np.array(
        [practice_learning_curve(result) for result in results], dtype=float
    )
    mean_prac_sat_vals, ster_prac_sat_vals = mean_ster(practice_curves)

    # amplifying-eigenvalue count per block kind
    amp_by_kind = [amplifying_eigenvalue_mean_by_kind(r) for r in results]
    mean_sat_by_kind, ster_sat_by_kind = mean_ster(stack_by_kind(amp_by_kind, kinds))

    # pooled eigenvalue magnitudes per block kind (all seeds, trials, units)
    eigmag_by_kind = []
    for kind in kinds:
        per_kind = [eigenvalue_magnitudes_by_kind(r)[kind] for r in results]
        eigmag_by_kind.append(np.concatenate(per_kind))

    # BEHAVIOURAL AGGREGATION #
    # ===================================================== #

    switch = [switch_contrast_by_kind(r) for r in results]
    incong = [incongruence_contrast_by_kind(r) for r in results]

    mean_switch_cost_rt, ster_switch_cost_rt = mean_ster(stack_by_kind(switch, kinds, 'rt_cost'))
    mean_switch_cost_acc, ster_switch_cost_acc = mean_ster(stack_by_kind(switch, kinds, 'accuracy_cost'))
    mean_incong_cost_rt, ster_incong_cost_rt = mean_ster(stack_by_kind(incong, kinds, 'rt_cost'))
    mean_incong_cost_acc, ster_incong_cost_acc = mean_ster(stack_by_kind(incong, kinds, 'accuracy_cost'))

    # pure (non-differenced) RT and accuracy per condition. hard = switch /
    # incongruent, easy = repeat / congruent
    mean_switch_rt, ster_switch_rt = mean_ster(stack_by_kind(switch, kinds, 'hard_rt'))
    mean_repeat_rt, ster_repeat_rt = mean_ster(stack_by_kind(switch, kinds, 'easy_rt'))
    mean_switch_acc, ster_switch_acc = mean_ster(stack_by_kind(switch, kinds, 'hard_accuracy'))
    mean_repeat_acc, ster_repeat_acc = mean_ster(stack_by_kind(switch, kinds, 'easy_accuracy'))
    mean_incongruent_rt, ster_incongruent_rt = mean_ster(stack_by_kind(incong, kinds, 'hard_rt'))
    mean_congruent_rt, ster_congruent_rt = mean_ster(stack_by_kind(incong, kinds, 'easy_rt'))
    mean_incongruent_acc, ster_incongruent_acc = mean_ster(stack_by_kind(incong, kinds, 'hard_accuracy'))
    mean_congruent_acc, ster_congruent_acc = mean_ster(stack_by_kind(incong, kinds, 'easy_accuracy'))

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

    # practice learning curve, with the boundary between the two practice blocks.
    # trials-per-block comes from the config (doubles under practice switching,
    # since a switching block mixes both rules)
    practice_trials = results[0].config.practice_trials
    prac_index = np.arange(0, practice_trials * num_practice_blocks, 1)
    fig = plt.figure()
    plt.errorbar(prac_index, mean_prac_sat_vals, yerr=ster_prac_sat_vals, color='black', capsize=3, linestyle='-', marker='o')
    plt.axvline(practice_trials - .5, color='grey', linestyle='--')
    plt.xlabel('Practice trial')
    plt.ylabel('Eigenvalues > 1')
    plt.xticks(prac_index[:: num_cues_per_rule * 4])
    fig.savefig(os.path.join(output_dir, 'practice_eigenvalues_above_1.png'), format='png', dpi=1200)

    # all eigenvalue magnitudes per block kind: individual pooled values
    # (jittered scatter) plus their mean +/- SE
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots()
    for i, values in enumerate(eigmag_by_kind):
        jitter = rng.uniform(-0.15, 0.15, size=values.shape)
        ax.scatter(np.full(values.shape, i) + jitter, values, s=4, alpha=0.15, color='grey', linewidths=0)
    eigmag_mean = [np.mean(values) for values in eigmag_by_kind]
    eigmag_ster = [np.std(values) / np.sqrt(values.size) for values in eigmag_by_kind]
    ax.errorbar(range(len(kinds)), eigmag_mean, yerr=eigmag_ster, color='black', capsize=3, linestyle='-', marker='o', zorder=3)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kind_labels)
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

    # overall RT and accuracy per block index, instruction/practice/real boundaries
    # marked. real blocks start after instruction blocks + one performance-practice
    # block per rule (see SwitchingExperimentConfig.include_performance_practice)
    num_practice_performance_blocks = len(PRACTICE_TASKS)
    real_blocks_start = num_practice_blocks + num_practice_performance_blocks
    num_blocks_total = real_blocks_start + len(switch_probs)
    block_index = np.arange(0, num_blocks_total, 1)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(block_index, mean_block_rt, yerr=ster_block_rt, color='black', capsize=3, linestyle='-', marker='o')
    ax[0].axvline(num_practice_blocks - .5, color='grey', linestyle='--')
    ax[0].axvline(real_blocks_start - .5, color='grey', linestyle='--')
    ax[0].set_xlabel('Block')
    ax[0].set_ylabel('RT (time-steps)')
    ax[0].set_xticks(block_index)
    ax[1].errorbar(block_index, mean_block_acc, yerr=ster_block_acc, color='black', capsize=3, linestyle='-', marker='o')
    ax[1].axvline(num_practice_blocks - .5, color='grey', linestyle='--')
    ax[1].axvline(real_blocks_start - .5, color='grey', linestyle='--')
    ax[1].set_xlabel('Block')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xticks(block_index)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'performance_evolution_by_block.png'), format='png', dpi=1200)

    # CONNECTIVITY SNAPSHOT #
    # ===================================================== #

    # W snapshot from one representative seed (final weights after all blocks).
    # attractor identity is arbitrary across seeds (Hebbian symmetry breaking from
    # the random weight init), so averaging W across seeds isn't meaningful
    snapshot_seed = 0
    W = results[snapshot_seed].final_combined_weights  # (num_features, num_conjunction_units)
    WWt = W @ W.T  # (num_features, num_features), the matrix behind feedback_eigenvalues

    vocabulary = build_vocabulary(num_cues_per_rule)
    feature_labels = [''] * vocabulary.number_of_features
    feature_labels[Feature.GREEN] = 'Green'
    feature_labels[Feature.BLUE] = 'Blue'
    feature_labels[Feature.SQUARE] = 'Square'
    feature_labels[Feature.CIRCLE] = 'Circle'
    for i, cue in enumerate(vocabulary.cues_by_task[Task.COLOR], start=1):
        feature_labels[cue] = f'Color cue {i}'
    for i, cue in enumerate(vocabulary.cues_by_task[Task.SHAPE], start=1):
        feature_labels[cue] = f'Shape cue {i}'
    feature_labels[vocabulary.response_features[0]] = 'Action 1'
    feature_labels[vocabulary.response_features[1]] = 'Action 2'

    # W heatmap, with a red box on each conjunction unit's most strongly bound
    # feature (the attractor association that unit has learned)
    fig, ax = plt.subplots(figsize=(4, 6))
    im = ax.imshow(W, aspect='auto', cmap='viridis')
    ax.set_yticks(range(vocabulary.number_of_features))
    ax.set_yticklabels(feature_labels)
    ax.set_xticks(range(number_of_conjunction_units))
    ax.set_xlabel('Conjunction unit')
    ax.set_title(f'W (seed {snapshot_seed}, final)')
    for j in range(number_of_conjunction_units):
        i = int(np.argmax(W[:, j]))
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, edgecolor='red', linewidth=2))
    fig.colorbar(im, ax=ax, label='Weight')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'connectivity_W_snapshot.png'), format='png', dpi=1200)

    # W @ W.T heatmap: the feedback matrix behind feedback_eigenvalues /
    # amplifying-eigenvalue counts
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(WWt, cmap='viridis')
    ax.set_yticks(range(vocabulary.number_of_features))
    ax.set_yticklabels(feature_labels)
    ax.set_xticks(range(vocabulary.number_of_features))
    ax.set_xticklabels(feature_labels, rotation=90)
    ax.set_title(f'W @ W.T (seed {snapshot_seed}, final)')
    fig.colorbar(im, ax=ax, label='Weight')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'connectivity_WWt_snapshot.png'), format='png', dpi=1200)

    # attractor path diagram: {stimulus, cue} -> conjunction unit -> action,
    # following each conjunction unit's most strongly bound stimulus, cue and
    # action (its learned rule mapping). stimulus and cue are both direct
    # inputs to the conjunction units (neither feeds through the other), so
    # they share the x=0 column and are told apart by line style. node
    # positions are normalized fractions per column so the layout stays
    # sensible regardless of num_cues_per_rule / number_of_conjunction_units
    stimulus_features = (Feature.GREEN, Feature.BLUE, Feature.SQUARE, Feature.CIRCLE)
    all_cues = vocabulary.cues_by_task[Task.COLOR] + vocabulary.cues_by_task[Task.SHAPE]
    all_inputs = stimulus_features + all_cues
    actions = vocabulary.response_features

    def frac_positions(items):
        n = len(items)
        return {item: (i / (n - 1) if n > 1 else 0.5) for i, item in enumerate(items)}

    input_y = frac_positions(all_inputs)
    conj_y = frac_positions(range(number_of_conjunction_units))
    action_y = frac_positions(actions)

    fig, ax = plt.subplots(figsize=(5, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, number_of_conjunction_units))
    for j in range(number_of_conjunction_units):
        top_stimulus = max(stimulus_features, key=lambda f: W[f, j])
        top_cue = max(all_cues, key=lambda f: W[f, j])
        top_action = max(actions, key=lambda f: W[f, j])
        ax.plot([0, 1], [input_y[top_stimulus], conj_y[j]], color=colors[j], linewidth=2,
                linestyle='dashed', marker='o')
        ax.plot([0, 1], [input_y[top_cue], conj_y[j]], color=colors[j], linewidth=2, marker='o')
        ax.plot([1, 2], [conj_y[j], action_y[top_action]], color=colors[j], linewidth=2, marker='o')

    for feature in all_inputs:
        ax.text(-0.05, input_y[feature], feature_labels[feature], ha='right', va='center')
    for j in range(number_of_conjunction_units):
        ax.text(1, conj_y[j] + 0.03, f'C{j}', ha='center', va='bottom')
    for action in actions:
        ax.text(2.05, action_y[action], feature_labels[action], ha='left', va='center')

    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Stimulus / Cue', 'Conjunction unit', 'Action'])
    ax.set_yticks([])
    ax.set_title(f'Learned attractor paths (seed {snapshot_seed}, final)')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'attractor_paths_snapshot.png'), format='png', dpi=1200)
    plt.close('all')


if __name__ == '__main__':
    if cpu_count >= 20:
        # enough headroom to also parallelise across model versions: one flat
        # pool of every (version, seed) job
        flat = [(v, cfg) for v in active_versions for cfg in build_configs(model_versions[v])]
        print(f'Running {len(flat)} seed jobs across {len(active_versions)} version(s) on {cpu_count} CPUs...')
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            flat_results = list(executor.map(run_switching_experiment, [cfg for _, cfg in flat]))
        results_by_version = {v: [] for v in active_versions}
        for (v, _), r in zip(flat, flat_results):
            results_by_version[v].append(r)
    else:
        # not enough CPUs to also split across versions: run versions one
        # after another, parallelising only their seeds
        results_by_version = {}
        for v in active_versions:
            configs = build_configs(model_versions[v])
            print(f'Running {n_seeds} seeds for {v}...')
            results_by_version[v] = run_seeds_parallel(configs, min(cpu_count, n_seeds))

    for version_name in active_versions:
        analyze_and_plot(version_name, results_by_version[version_name])
