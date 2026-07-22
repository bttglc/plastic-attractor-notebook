"""Run the cued task-switching experiment across seeds and plot the results.

OOP counterpart of the flat simulation_launcher.py (which stays in place). It
runs the experiment for each seed, reduces the trials with gated_attractor's
analysis functions, averages across seeds, and writes the same figures.

Run it from the updated_model folder:

    python gated_attractor/launcher.py
"""

import argparse
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np

# In launcher.py, after cpu_count = os.cpu_count() or 1
cpu_count = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))

# make the gated_attractor package (this file's parent dir) importable regardless of the cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gated_attractor import (
    PRACTICE_TASKS,
    SwitchingExperimentConfig,
    amplifying_eigenvalue_mean_by_kind,
    block_kinds,
    conjunction_routing_drift_by_kind,
    conjunction_routing_flip_rate_by_block,
    conjunction_routing_flip_rate_full_session,
    eigenvalue_magnitudes_by_kind,
    gate_accuracy_by_block,
    incongruence_contrast_by_kind,
    no_response_rate_by_block,
    performance_by_block,
    practice_learning_curve,
    relevant_irrelevant_activity_by_kind,
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
practice_permutation_repeats = 5

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
active_versions = [
    '2cpr_slowW3'
]

# PARALLEL SEED EXECUTION #
# ===================================================== #


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


def run_seeds_parallel(configs, max_workers, label=''):
    # executor.map preserves input order, so results[i] still corresponds to
    # seed=i (needed downstream, e.g. snapshot_seed = 0)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = []
        for i, result in enumerate(executor.map(run_switching_experiment, configs), start=1):
            print(f'  {label}seed {i}/{len(configs)} done')
            results.append(result)
        return results


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

    # read the unit count off the actual run config rather than the shared
    # model_versions_config default, so plotting stays correct even if a
    # version entry overrides it
    params = results[0].config.model_parameters
    num_cues_per_rule = params.num_cues_per_rule

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
    amp_by_kind_raw = stack_by_kind(amp_by_kind, kinds)
    mean_sat_by_kind, ster_sat_by_kind = mean_ster(amp_by_kind_raw)

    # pooled eigenvalue magnitudes per block kind, kept per-seed (n_kinds, n_seeds)
    # ragged object array for reuse, and also concatenated across seeds for the
    # scatter plot below (all seeds, trials, units)
    eigmag_by_kind_per_seed = np.empty((len(kinds), len(results)), dtype=object)
    eigmag_by_kind = []
    for k, kind in enumerate(kinds):
        per_kind = [eigenvalue_magnitudes_by_kind(r)[kind] for r in results]
        eigmag_by_kind_per_seed[k, :] = per_kind
        eigmag_by_kind.append(np.concatenate(per_kind))

    # BEHAVIOURAL AGGREGATION #
    # ===================================================== #

    switch = [switch_contrast_by_kind(r) for r in results]
    incong = [incongruence_contrast_by_kind(r) for r in results]

    switch_rt_cost_raw = stack_by_kind(switch, kinds, 'rt_cost')
    switch_accuracy_cost_raw = stack_by_kind(switch, kinds, 'accuracy_cost')
    incong_rt_cost_raw = stack_by_kind(incong, kinds, 'rt_cost')
    incong_accuracy_cost_raw = stack_by_kind(incong, kinds, 'accuracy_cost')
    mean_switch_cost_rt, ster_switch_cost_rt = mean_ster(switch_rt_cost_raw)
    mean_switch_cost_acc, ster_switch_cost_acc = mean_ster(switch_accuracy_cost_raw)
    mean_incong_cost_rt, ster_incong_cost_rt = mean_ster(incong_rt_cost_raw)
    mean_incong_cost_acc, ster_incong_cost_acc = mean_ster(incong_accuracy_cost_raw)

    # pure (non-differenced) RT and accuracy per condition. hard = switch /
    # incongruent, easy = repeat / congruent
    switch_hard_rt_raw = stack_by_kind(switch, kinds, 'hard_rt')
    switch_easy_rt_raw = stack_by_kind(switch, kinds, 'easy_rt')
    switch_hard_accuracy_raw = stack_by_kind(switch, kinds, 'hard_accuracy')
    switch_easy_accuracy_raw = stack_by_kind(switch, kinds, 'easy_accuracy')
    incong_hard_rt_raw = stack_by_kind(incong, kinds, 'hard_rt')
    incong_easy_rt_raw = stack_by_kind(incong, kinds, 'easy_rt')
    incong_hard_accuracy_raw = stack_by_kind(incong, kinds, 'hard_accuracy')
    incong_easy_accuracy_raw = stack_by_kind(incong, kinds, 'easy_accuracy')
    mean_switch_rt, ster_switch_rt = mean_ster(switch_hard_rt_raw)
    mean_repeat_rt, ster_repeat_rt = mean_ster(switch_easy_rt_raw)
    mean_switch_acc, ster_switch_acc = mean_ster(switch_hard_accuracy_raw)
    mean_repeat_acc, ster_repeat_acc = mean_ster(switch_easy_accuracy_raw)
    mean_incongruent_rt, ster_incongruent_rt = mean_ster(incong_hard_rt_raw)
    mean_congruent_rt, ster_congruent_rt = mean_ster(incong_easy_rt_raw)
    mean_incongruent_acc, ster_incongruent_acc = mean_ster(incong_hard_accuracy_raw)
    mean_congruent_acc, ster_congruent_acc = mean_ster(incong_easy_accuracy_raw)

    # PERFORMANCE EVOLUTION #
    # ===================================================== #

    perf = [performance_by_block(r) for r in results]
    block_rt = np.array([rt for rt, _ in perf], dtype=float)
    block_acc = np.array([acc for _, acc in perf], dtype=float)
    mean_block_rt, ster_block_rt = mean_ster(block_rt)
    mean_block_acc, ster_block_acc = mean_ster(block_acc)

    # ROOT-CAUSE DIAGNOSTICS #
    # ===================================================== #

    gate_acc_by_block = np.array(
        [gate_accuracy_by_block(r) for r in results], dtype=float
    )
    mean_gate_acc_by_block, ster_gate_acc_by_block = mean_ster(gate_acc_by_block)

    no_response_by_block = np.array(
        [no_response_rate_by_block(r) for r in results], dtype=float
    )
    mean_no_response_by_block, ster_no_response_by_block = mean_ster(no_response_by_block)

    activity = [relevant_irrelevant_activity_by_kind(r) for r in results]
    relevant_activity_raw = stack_by_kind(activity, kinds, 'relevant')
    irrelevant_activity_raw = stack_by_kind(activity, kinds, 'irrelevant')
    mean_relevant_activity, ster_relevant_activity = mean_ster(relevant_activity_raw)
    mean_irrelevant_activity, ster_irrelevant_activity = mean_ster(irrelevant_activity_raw)

    # does the settled winner-take-all conjunction unit for a given (task,
    # stimulus) stay the same trial to trial (within a block) and block to
    # block (within a kind), or does routing drift under continuous
    # real-block learning? see model_outline.md section 13
    routing_flip_by_block = np.array(
        [conjunction_routing_flip_rate_by_block(r) for r in results], dtype=float
    )
    mean_routing_flip_by_block, ster_routing_flip_by_block = mean_ster(routing_flip_by_block)

    routing_drift = [conjunction_routing_drift_by_kind(r) for r in results]
    routing_drift_raw = stack_by_kind(routing_drift, kinds)
    mean_routing_drift_by_kind, ster_routing_drift_by_kind = mean_ster(routing_drift_raw)

    # whole-session version of the same question, pooling all 8 real blocks
    # instead of just the ~2 sharing a switch-probability kind -- directly
    # comparable to model_outline.md section 13's "0/96" headline number
    routing_flip_full_session_raw = np.array(
        [conjunction_routing_flip_rate_full_session(r) for r in results], dtype=float
    )
    mean_routing_flip_full_session, ster_routing_flip_full_session = mean_ster(
        routing_flip_full_session_raw
    )

    # RAW DATA EXPORT #
    # ===================================================== #

    # final combined weights per seed, for reuse without rerunning simulations
    W_by_seed = np.array([r.final_combined_weights for r in results])

    np.savez(
        os.path.join(output_dir, 'simulation_data.npz'),
        kinds=np.array(kinds, dtype=float),
        block_index=np.arange(0, block_rt.shape[1]),
        prac_index=np.arange(0, practice_curves.shape[1]),
        # raw per-seed arrays
        practice_curves=practice_curves,
        amp_by_kind_raw=amp_by_kind_raw,
        eigmag_by_kind_per_seed=eigmag_by_kind_per_seed,
        switch_rt_cost_raw=switch_rt_cost_raw,
        switch_accuracy_cost_raw=switch_accuracy_cost_raw,
        switch_hard_rt_raw=switch_hard_rt_raw,
        switch_easy_rt_raw=switch_easy_rt_raw,
        switch_hard_accuracy_raw=switch_hard_accuracy_raw,
        switch_easy_accuracy_raw=switch_easy_accuracy_raw,
        incong_rt_cost_raw=incong_rt_cost_raw,
        incong_accuracy_cost_raw=incong_accuracy_cost_raw,
        incong_hard_rt_raw=incong_hard_rt_raw,
        incong_easy_rt_raw=incong_easy_rt_raw,
        incong_hard_accuracy_raw=incong_hard_accuracy_raw,
        incong_easy_accuracy_raw=incong_easy_accuracy_raw,
        block_rt=block_rt,
        block_acc=block_acc,
        gate_acc_by_block=gate_acc_by_block,
        no_response_by_block=no_response_by_block,
        relevant_activity_raw=relevant_activity_raw,
        irrelevant_activity_raw=irrelevant_activity_raw,
        routing_flip_by_block=routing_flip_by_block,
        routing_drift_raw=routing_drift_raw,
        routing_flip_full_session_raw=routing_flip_full_session_raw,
        W_by_seed=W_by_seed,
        # aggregated mean/standard-error arrays (as plotted)
        mean_prac_sat_vals=mean_prac_sat_vals, ster_prac_sat_vals=ster_prac_sat_vals,
        mean_sat_by_kind=mean_sat_by_kind, ster_sat_by_kind=ster_sat_by_kind,
        mean_switch_cost_rt=mean_switch_cost_rt, ster_switch_cost_rt=ster_switch_cost_rt,
        mean_switch_cost_acc=mean_switch_cost_acc, ster_switch_cost_acc=ster_switch_cost_acc,
        mean_incong_cost_rt=mean_incong_cost_rt, ster_incong_cost_rt=ster_incong_cost_rt,
        mean_incong_cost_acc=mean_incong_cost_acc, ster_incong_cost_acc=ster_incong_cost_acc,
        mean_switch_rt=mean_switch_rt, ster_switch_rt=ster_switch_rt,
        mean_repeat_rt=mean_repeat_rt, ster_repeat_rt=ster_repeat_rt,
        mean_switch_acc=mean_switch_acc, ster_switch_acc=ster_switch_acc,
        mean_repeat_acc=mean_repeat_acc, ster_repeat_acc=ster_repeat_acc,
        mean_incongruent_rt=mean_incongruent_rt, ster_incongruent_rt=ster_incongruent_rt,
        mean_congruent_rt=mean_congruent_rt, ster_congruent_rt=ster_congruent_rt,
        mean_incongruent_acc=mean_incongruent_acc, ster_incongruent_acc=ster_incongruent_acc,
        mean_congruent_acc=mean_congruent_acc, ster_congruent_acc=ster_congruent_acc,
        mean_block_rt=mean_block_rt, ster_block_rt=ster_block_rt,
        mean_block_acc=mean_block_acc, ster_block_acc=ster_block_acc,
        mean_gate_acc_by_block=mean_gate_acc_by_block, ster_gate_acc_by_block=ster_gate_acc_by_block,
        mean_no_response_by_block=mean_no_response_by_block, ster_no_response_by_block=ster_no_response_by_block,
        mean_relevant_activity=mean_relevant_activity, ster_relevant_activity=ster_relevant_activity,
        mean_irrelevant_activity=mean_irrelevant_activity, ster_irrelevant_activity=ster_irrelevant_activity,
        mean_routing_flip_by_block=mean_routing_flip_by_block, ster_routing_flip_by_block=ster_routing_flip_by_block,
        mean_routing_drift_by_kind=mean_routing_drift_by_kind, ster_routing_drift_by_kind=ster_routing_drift_by_kind,
        mean_routing_flip_full_session=mean_routing_flip_full_session, ster_routing_flip_full_session=ster_routing_flip_full_session,
    )

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

    # relevant vs irrelevant feature activity (settled, end of response_window)
    # per block kind -- is the suppressed/driven signal itself stable across
    # switch-probability conditions, or degraded under harder ones?
    fig, ax = plt.subplots()
    ax.errorbar(kinds, mean_relevant_activity, yerr=ster_relevant_activity, color='black', capsize=3, linestyle='-', marker='o', label='Relevant')
    ax.errorbar(kinds, mean_irrelevant_activity, yerr=ster_irrelevant_activity, color='black', capsize=3, linestyle='--', marker='s', label='Irrelevant')
    ax.set_xlabel('Switch probability (block kind)')
    ax.set_ylabel('Settled feature activity')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'relevant_irrelevant_activity_by_kind.png'), format='png', dpi=1200)

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

    # root-cause diagnostics per block index, same instruction/real boundaries:
    # is the gate itself ever wrong, and does the network fail to respond at
    # all (vs. respond incorrectly)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(block_index, mean_gate_acc_by_block, yerr=ster_gate_acc_by_block, color='black', capsize=3, linestyle='-', marker='o')
    ax[0].axvline(num_practice_blocks - .5, color='grey', linestyle='--')
    ax[0].axvline(real_blocks_start - .5, color='grey', linestyle='--')
    ax[0].set_xlabel('Block')
    ax[0].set_ylabel('Gate accuracy')
    ax[0].set_xticks(block_index)
    ax[1].errorbar(block_index, mean_no_response_by_block, yerr=ster_no_response_by_block, color='black', capsize=3, linestyle='-', marker='o')
    ax[1].axvline(num_practice_blocks - .5, color='grey', linestyle='--')
    ax[1].axvline(real_blocks_start - .5, color='grey', linestyle='--')
    ax[1].set_xlabel('Block')
    ax[1].set_ylabel('No-response rate')
    ax[1].set_xticks(block_index)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'diagnostics_by_block.png'), format='png', dpi=1200)

    # conjunction-unit routing stability: does the same physical (task,
    # stimulus) keep winning settled winner-take-all on the same conjunction
    # unit, within a block (left) and across the blocks sharing a switch
    # probability (right)? see model_outline.md section 13 -- real-block
    # learning never turns off, so nothing anchors this by construction
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].errorbar(block_index, mean_routing_flip_by_block, yerr=ster_routing_flip_by_block, color='black', capsize=3, linestyle='-', marker='o')
    ax[0].axvline(num_practice_blocks - .5, color='grey', linestyle='--')
    ax[0].axvline(real_blocks_start - .5, color='grey', linestyle='--')
    ax[0].set_xlabel('Block')
    ax[0].set_ylabel('Within-block routing flip rate')
    ax[0].set_xticks(block_index)
    ax[1].errorbar(kinds, mean_routing_drift_by_kind, yerr=ster_routing_drift_by_kind, color='black', capsize=3, linestyle='-', marker='o')
    ax[1].set_xlabel('Switch probability (block kind)')
    ax[1].set_ylabel('Cross-block routing drift rate')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'conjunction_routing_stability.png'), format='png', dpi=1200)

    plt.close('all')

def save_results_summary(version_name, results, output_dir):
    """
    Save simulation results in wide format for easy comparison.
    
    Creates:
    1. {version}_summary.csv - Single row with all summary metrics
    2. {version}_detailed_blocks.csv - Per-block metrics (wide format)
    3. {version}_detailed_kinds.csv - Per-kind (switch probability) metrics (wide format)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    first_result = results[0]
    params = first_result.config.model_parameters
    config = first_result.config
    
    # ---- compute all metrics (same as analyze_and_plot) ----
    kinds = block_kinds(first_result)
    kind_labels = [str(kind) for kind in kinds]
    
    # Practice curves
    practice_curves = np.array([practice_learning_curve(r) for r in results], dtype=float)
    mean_prac, ster_prac = mean_ster(practice_curves)
    
    # Amplifying eigenvalues
    amp_by_kind = [amplifying_eigenvalue_mean_by_kind(r) for r in results]
    amp_raw = stack_by_kind(amp_by_kind, kinds)
    mean_amp, ster_amp = mean_ster(amp_raw)
    
    # Behavioural metrics
    switch = [switch_contrast_by_kind(r) for r in results]
    incong = [incongruence_contrast_by_kind(r) for r in results]
    
    switch_rt_raw = stack_by_kind(switch, kinds, 'rt_cost')
    switch_acc_raw = stack_by_kind(switch, kinds, 'accuracy_cost')
    incong_rt_raw = stack_by_kind(incong, kinds, 'rt_cost')
    incong_acc_raw = stack_by_kind(incong, kinds, 'accuracy_cost')
    
    mean_switch_rt, ster_switch_rt = mean_ster(switch_rt_raw)
    mean_switch_acc, ster_switch_acc = mean_ster(switch_acc_raw)
    mean_incong_rt, ster_incong_rt = mean_ster(incong_rt_raw)
    mean_incong_acc, ster_incong_acc = mean_ster(incong_acc_raw)
    
    # Raw RT and accuracy per condition
    switch_hard_rt = stack_by_kind(switch, kinds, 'hard_rt')
    switch_easy_rt = stack_by_kind(switch, kinds, 'easy_rt')
    switch_hard_acc = stack_by_kind(switch, kinds, 'hard_accuracy')
    switch_easy_acc = stack_by_kind(switch, kinds, 'easy_accuracy')
    
    mean_switch_rt_raw, _ = mean_ster(switch_hard_rt)          # switch RT
    mean_repeat_rt_raw, _ = mean_ster(switch_easy_rt)          # repeat RT
    mean_switch_acc_raw, _ = mean_ster(switch_hard_acc)        # switch accuracy
    mean_repeat_acc_raw, _ = mean_ster(switch_easy_acc)        # repeat accuracy
    
    # Per-kind congruent/incongruent (from incong contrast)
    incong_hard_rt = stack_by_kind(incong, kinds, 'hard_rt')       # incongruent RT
    incong_easy_rt = stack_by_kind(incong, kinds, 'easy_rt')       # congruent RT
    incong_hard_acc = stack_by_kind(incong, kinds, 'hard_accuracy') # incongruent acc
    incong_easy_acc = stack_by_kind(incong, kinds, 'easy_accuracy') # congruent acc
    
    mean_incongruent_rt, ster_incongruent_rt = mean_ster(incong_hard_rt)
    mean_congruent_rt, ster_congruent_rt = mean_ster(incong_easy_rt)
    mean_incongruent_acc, ster_incongruent_acc = mean_ster(incong_hard_acc)
    mean_congruent_acc, ster_congruent_acc = mean_ster(incong_easy_acc)
    
    # Overall performance (all blocks)
    perf = [performance_by_block(r) for r in results]
    block_rt = np.array([rt for rt, _ in perf], dtype=float)
    block_acc = np.array([acc for _, acc in perf], dtype=float)
    mean_block_rt, ster_block_rt = mean_ster(block_rt)
    mean_block_acc, ster_block_acc = mean_ster(block_acc)
    
    # ---- Real‑blocks accuracy (exclude practice & instruction) ----
    num_practice_performance_blocks = len(PRACTICE_TASKS)
    real_blocks_start = num_practice_blocks + num_practice_performance_blocks
    # block_acc shape: (n_seeds, num_blocks)
    real_block_acc = block_acc[:, real_blocks_start:]   # select columns from real_blocks_start onward
    mean_real_block_acc_per_seed = np.mean(real_block_acc, axis=1)  # mean over real blocks for each seed
    mean_real_blocks_acc, ster_real_blocks_acc = mean_ster(mean_real_block_acc_per_seed[:, None])  # to get scalar
    
    # No-response rate (all blocks)
    no_response = np.array([no_response_rate_by_block(r) for r in results], dtype=float)
    mean_no_response, ster_no_response = mean_ster(no_response)
    
    # Routing stability
    routing_flip = np.array([conjunction_routing_flip_rate_by_block(r) for r in results], dtype=float)
    mean_routing_flip, ster_routing_flip = mean_ster(routing_flip)
    
    routing_drift = [conjunction_routing_drift_by_kind(r) for r in results]
    routing_drift_raw = stack_by_kind(routing_drift, kinds)
    mean_routing_drift, ster_routing_drift = mean_ster(routing_drift_raw)
    
    routing_flip_full = np.array([conjunction_routing_flip_rate_full_session(r) for r in results], dtype=float)
    mean_routing_flip_full, ster_routing_flip_full = mean_ster(routing_flip_full)
    
    # =============================================
    # 1. SUMMARY FILE - One row per version
    # =============================================
    summary_row = {
        'version': version_name,
        'timestamp': datetime.now().isoformat(),
        'n_seeds': len(results),
        'num_trials': config.num_trials,
        'num_practice_blocks': config.num_practice_blocks,
        'practice_permutation_repeats': config.practice_permutation_repeats,
        'practice_switch_probability': config.practice_switch_probability,
        'num_cues_per_rule': params.num_cues_per_rule,
        'num_conjunction_units': params.number_of_conjunction_units,
        'learning_rate': getattr(params, 'learning_rate', None),
        'decay_rate': getattr(params, 'decay_rate', None),
        'mean_practice_final': float(mean_prac[-1]) if len(mean_prac) > 0 else None,
        'mean_amplifying_eigenvalues': float(np.mean(mean_amp)),
        'switch_cost_rt_mean': float(np.mean(mean_switch_rt)),
        'switch_cost_acc_mean': float(np.mean(mean_switch_acc)),
        'incong_cost_rt_mean': float(np.mean(mean_incong_rt)),
        'incong_cost_acc_mean': float(np.mean(mean_incong_acc)),
        'overall_rt': float(np.mean(mean_block_rt)),
        'overall_accuracy': float(np.mean(mean_block_acc)),           # includes practice
        'real_blocks_accuracy': float(mean_real_blocks_acc[0]),        # new: only real blocks
        'no_response_rate': float(np.mean(mean_no_response)),
        'routing_flip_full_session': float(mean_routing_flip_full),
        'routing_drift_mean': float(np.mean(mean_routing_drift)),
    }
    
    # Add per-kind metrics as columns
    for i, label in enumerate(kind_labels):
        label_str = str(label).replace('.', '_')
        summary_row[f'switch_cost_rt_{label_str}'] = float(mean_switch_rt[i])
        summary_row[f'switch_cost_acc_{label_str}'] = float(mean_switch_acc[i])
        summary_row[f'incong_cost_rt_{label_str}'] = float(mean_incong_rt[i])
        summary_row[f'incong_cost_acc_{label_str}'] = float(mean_incong_acc[i])
        summary_row[f'amp_eigenvalues_{label_str}'] = float(mean_amp[i])
        summary_row[f'routing_drift_{label_str}'] = float(mean_routing_drift[i])
        # New per-kind congruent/incongruent RT and accuracy
        summary_row[f'congruent_rt_{label_str}'] = float(mean_congruent_rt[i])
        summary_row[f'incongruent_rt_{label_str}'] = float(mean_incongruent_rt[i])
        summary_row[f'congruent_acc_{label_str}'] = float(mean_congruent_acc[i])
        summary_row[f'incongruent_acc_{label_str}'] = float(mean_incongruent_acc[i])
    
    # Save summary CSV
    summary_path = os.path.join(output_dir, f'{version_name}_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_row.keys())
        writer.writeheader()
        writer.writerow(summary_row)
    print(f"  Saved summary to: {summary_path}")
    
    # =============================================
    # 2. BLOCK-LEVEL DETAILS - One row per block
    # =============================================
    num_blocks = len(mean_block_rt)
    block_data = []
    for block_idx in range(num_blocks):
        block_row = {
            'version': version_name,
            'block_index': block_idx,
            'rt_mean': float(mean_block_rt[block_idx]),
            'rt_ster': float(ster_block_rt[block_idx]) if ster_block_rt[block_idx] is not None else None,
            'accuracy_mean': float(mean_block_acc[block_idx]),
            'accuracy_ster': float(ster_block_acc[block_idx]) if ster_block_acc[block_idx] is not None else None,
            'no_response_mean': float(mean_no_response[block_idx]),
            'no_response_ster': float(ster_no_response[block_idx]) if ster_no_response[block_idx] is not None else None,
            'routing_flip_mean': float(mean_routing_flip[block_idx]),
            'routing_flip_ster': float(ster_routing_flip[block_idx]) if ster_routing_flip[block_idx] is not None else None,
        }
        block_data.append(block_row)
    
    block_path = os.path.join(output_dir, f'{version_name}_detailed_blocks.csv')
    with open(block_path, 'w', newline='') as f:
        if block_data:
            writer = csv.DictWriter(f, fieldnames=block_data[0].keys())
            writer.writeheader()
            writer.writerows(block_data)
    print(f"  Saved block details to: {block_path}")
    
    # =============================================
    # 3. KIND-LEVEL DETAILS - One row per kind
    # =============================================
    kind_data = []
    for i, label in enumerate(kind_labels):
        label_str = str(label)
        kind_row = {
            'version': version_name,
            'switch_probability': label_str,
            # switch costs
            'switch_cost_rt_mean': float(mean_switch_rt[i]),
            'switch_cost_rt_ster': float(ster_switch_rt[i]),
            'switch_cost_acc_mean': float(mean_switch_acc[i]),
            'switch_cost_acc_ster': float(ster_switch_acc[i]),
            # incongruence costs
            'incong_cost_rt_mean': float(mean_incong_rt[i]),
            'incong_cost_rt_ster': float(ster_incong_rt[i]),
            'incong_cost_acc_mean': float(mean_incong_acc[i]),
            'incong_cost_acc_ster': float(ster_incong_acc[i]),
            # eigenvalue
            'amp_eigenvalues_mean': float(mean_amp[i]),
            'amp_eigenvalues_ster': float(ster_amp[i]),
            # routing drift
            'routing_drift_mean': float(mean_routing_drift[i]),
            'routing_drift_ster': float(ster_routing_drift[i]),
            # NEW: congruent/incongruent RT and accuracy
            'congruent_rt_mean': float(mean_congruent_rt[i]),
            'congruent_rt_ster': float(ster_congruent_rt[i]),
            'incongruent_rt_mean': float(mean_incongruent_rt[i]),
            'incongruent_rt_ster': float(ster_incongruent_rt[i]),
            'congruent_acc_mean': float(mean_congruent_acc[i]),
            'congruent_acc_ster': float(ster_congruent_acc[i]),
            'incongruent_acc_mean': float(mean_incongruent_acc[i]),
            'incongruent_acc_ster': float(ster_incongruent_acc[i]),
        }
        kind_data.append(kind_row)
    
    kind_path = os.path.join(output_dir, f'{version_name}_detailed_kinds.csv')
    with open(kind_path, 'w', newline='') as f:
        if kind_data:
            writer = csv.DictWriter(f, fieldnames=kind_data[0].keys())
            writer.writeheader()
            writer.writerows(kind_data)
    print(f"  Saved kind details to: {kind_path}")

def append_to_master_log(master_csv_path, version_name, output_dir):
    """
    Append the summary CSV from a single run to a master CSV file containing
    all runs from the parameter sweep.
    """
    summary_csv = os.path.join(output_dir, 'summary.csv')
    
    if not os.path.exists(summary_csv):
        print(f"Warning: {summary_csv} not found")
        return
    
    # Read the summary CSV
    df = pd.read_csv(summary_csv)
    
    # Append to master CSV
    if os.path.exists(master_csv_path):
        # Append without header
        df.to_csv(master_csv_path, mode='a', header=False, index=False)
    else:
        # Write with header
        df.to_csv(master_csv_path, index=False)
    
    print(f"  Appended to master log: {master_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Run cued task-switching experiment')
    parser.add_argument('--version', type=str, required=True,
                        help='Model version name from model_versions_config')
    parser.add_argument('--seeds', type=int, default=20,
                        help='Number of random seeds to run')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting (save time on HPC)')
    parser.add_argument('--no-logs', action='store_true',
                        help='Skip detailed logs')
    parser.add_argument('--num-trials', type=int, default=None,
                        help='Override number of trials per block')
    parser.add_argument('--practice-repeats', type=int, default=None,
                        help='Override practice permutation repeats')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Override globals if provided
    if args.seeds:
        n_seeds = args.seeds
    if args.num_trials is not None:
        num_trials = args.num_trials
    if args.practice_repeats is not None:
        practice_permutation_repeats = args.practice_repeats
    
    # Only run the specified version
    active_versions = [args.version]  # Override the global list
    
    # Check if version exists
    if args.version not in model_versions:
        print(f"Error: Version '{args.version}' not found in model_versions_config")
        print(f"Available versions: {list(model_versions.keys())}")
        sys.exit(1)
    
    # Override output directory if provided
    if args.output_dir is not None:
        base_output_dir = args.output_dir
    
    # Simplified execution for single version
    configs = build_configs(model_versions[args.version])
    print(f'Running {n_seeds} seeds for {args.version}...')
    results = run_seeds_parallel(configs, min(cpu_count, n_seeds), label=f'{args.version}: ')
    
    # Optional: Skip plotting on HPC to save time/disk
    if not args.no_plot:
        analyze_and_plot(args.version, results)
    
    # Optional: Skip detailed logs if running many jobs
    if not args.no_logs:
        output_dir = os.path.join(base_output_dir, args.version)
        save_results_summary(args.version, results, output_dir) 
            
    print(f'Done: {args.version} complete.')