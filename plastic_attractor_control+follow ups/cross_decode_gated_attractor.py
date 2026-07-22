"""Cross-cue decoding: train a decoder on trials using one cue, test on
trials using the OTHER cue for the same rule (e.g. train on cueA1 trials,
test on cueA2 trials, both "attend colour"). If the relevant dimension
still decodes well across that split, that's representational-level
evidence of a cue-general code -- the same logic as the behavioural
novel-cue control (Control 2), but read directly off population activity
instead of accuracy.

Reuses decode_gated_attractor.py's population/label helpers; the only
change is the train/test split, from held-out block to held-out cue
identity. Restricted to the plateau windows already established in
decode_gated_attractor_stats.py (200-400 for conjunction units, 150-340
for feature units) rather than the full 0-400 trial -- chosen there, not
re-picked here, and it also keeps this cheap enough to run at n=20.

Run it from the project root:

    python cross_decode_gated_attractor.py
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.getcwd()
sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.join(_this_dir, 'gated_attractor'))

from decode_gated_attractor import (
    _population_activity, _stimulus_label, model_parameters, num_trials,
    practice_permutation_repeats, switch_probs,
)
from gated_attractor import (
    SwitchingExperimentConfig, Task, build_vocabulary, cues_for_task,
    run_switching_experiment,
)

N_SEEDS = 20
PLATEAU_WINDOWS = {
    'conjunction': (200, 400),
    'feature': (150, 340),
}

# real cue indices for this preset, not hardcoded -- same pattern
# gated_attractor/run_controls.py itself uses
_vocabulary = build_vocabulary(model_parameters.num_cues_per_rule)
CUES_BY_TASK = {
    Task.COLOR: cues_for_task(_vocabulary, Task.COLOR),
    Task.SHAPE: cues_for_task(_vocabulary, Task.SHAPE),
}


def paired_comparison(a, b, equivalence_bound=0.05, alpha=0.05):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    se_diff = sd_diff / np.sqrt(n) if sd_diff > 0 else 0.0
    t_stat, p_value = ttest_rel(a, b)
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff
    cohens_d = mean_diff / sd_diff if sd_diff > 0 else float('nan')
    if se_diff > 0:
        t_upper = (mean_diff - equivalence_bound) / se_diff
        p_upper = scipy_stats.t.cdf(t_upper, df=n - 1)
        t_lower = (mean_diff + equivalence_bound) / se_diff
        p_lower = 1 - scipy_stats.t.cdf(t_lower, df=n - 1)
        tost_p = max(p_lower, p_upper)
    else:
        tost_p = 0.0 if abs(mean_diff) < equivalence_bound else 1.0
    equivalent = tost_p < alpha
    return {'mean_diff': mean_diff, 'ci_low': ci_low, 'ci_high': ci_high,
            'cohens_d': cohens_d, 't': float(t_stat), 'p': float(p_value),
            'tost_p': tost_p, 'equivalent': equivalent}


def _fit_test(train_trials, test_trials, dimension, population, time_steps):
    train_labels = np.array([_stimulus_label(t, dimension) for t in train_trials])
    test_labels = np.array([_stimulus_label(t, dimension) for t in test_trials])
    if np.unique(train_labels).size < 2 or np.unique(test_labels).size < 2:
        return np.full(len(time_steps), np.nan)

    train_activity = np.stack([_population_activity(t, population) for t in train_trials])
    test_activity = np.stack([_population_activity(t, population) for t in test_trials])

    acc = np.empty(len(time_steps))
    for i, step in enumerate(time_steps):
        decoder = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
        decoder.fit(train_activity[:, step], train_labels)
        predictions = decoder.predict(test_activity[:, step])
        acc[i] = balanced_accuracy_score(test_labels, predictions)
    return acc


def cross_cue_decode(trials, *, task, decoded_dimension, population, time_steps):
    """Average of train-on-cue-A/test-on-cue-B and the reverse direction,
    for the two cues belonging to `task`."""

    cue_a, cue_b = CUES_BY_TASK[task]

    def trials_for(cue):
        return [
            t for t in trials
            if t.correct and t.task == task and not t.is_practice and t.cue == cue
        ]

    trials_a, trials_b = trials_for(cue_a), trials_for(cue_b)
    if not trials_a or not trials_b:
        return np.full(len(time_steps), np.nan)

    acc_ab = _fit_test(trials_a, trials_b, decoded_dimension, population, time_steps)
    acc_ba = _fit_test(trials_b, trials_a, decoded_dimension, population, time_steps)
    return np.nanmean([acc_ab, acc_ba], axis=0)


def cross_cue_decode_by_relevance(trials, *, population, time_steps):
    relevant_curves, irrelevant_curves = [], []
    for task in Task:
        irrelevant_dimension = Task.SHAPE if task == Task.COLOR else Task.COLOR
        relevant_curves.append(
            cross_cue_decode(trials, task=task, decoded_dimension=task,
                              population=population, time_steps=time_steps)
        )
        irrelevant_curves.append(
            cross_cue_decode(trials, task=task, decoded_dimension=irrelevant_dimension,
                              population=population, time_steps=time_steps)
        )
    return np.nanmean(relevant_curves, axis=0), np.nanmean(irrelevant_curves, axis=0)


def run_one_seed(seed):
    cfg = SwitchingExperimentConfig(
        seed=seed, num_trials=num_trials,
        practice_permutation_repeats=practice_permutation_repeats,
        switch_probs=switch_probs, model_parameters=model_parameters,
    )
    trials = run_switching_experiment(cfg).trials

    plateau_means = {}
    for population, (win_start, win_stop) in PLATEAU_WINDOWS.items():
        time_steps = np.arange(win_start, win_stop, 5)
        relevant, irrelevant = cross_cue_decode_by_relevance(
            trials, population=population, time_steps=time_steps,
        )
        plateau_means[population] = (float(np.nanmean(relevant)), float(np.nanmean(irrelevant)))
    return plateau_means


if __name__ == '__main__':
    print(f'Cross-cue decoding across {N_SEEDS} seeds (2cpr_slowW3 preset)...')
    with ProcessPoolExecutor() as ex:
        results = []
        for i, r in enumerate(ex.map(run_one_seed, range(N_SEEDS)), start=1):
            results.append(r)
            print(f'  seed {i}/{N_SEEDS} done')

    print(f'\n=== Cross-cue decoding (train on one cue, test on the other), n={N_SEEDS} ===')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, population in zip(axes, ('conjunction', 'feature')):
        relevant = np.array([r[population][0] for r in results])
        irrelevant = np.array([r[population][1] for r in results])
        chance = np.full(N_SEEDS, 0.5)

        rel_vs_chance = paired_comparison(relevant, chance)
        irr_vs_chance = paired_comparison(irrelevant, chance)

        print(f'\n--- {population} units ---')
        print(f'relevant (cross-cue):   {relevant.mean():.3f} (sd={relevant.std(ddof=1):.3f})')
        print(f'irrelevant (cross-cue): {irrelevant.mean():.3f} (sd={irrelevant.std(ddof=1):.3f})')
        rel_verdict = 'EQUIVALENT to chance -> no cross-cue generalisation' if rel_vs_chance['equivalent'] else 'not equivalent to chance'
        print(f"relevant vs chance (0.5): t={rel_vs_chance['t']:.3f}, p={rel_vs_chance['p']:.5f}, "
              f"TOST (bound=+/-0.05): p={rel_vs_chance['tost_p']:.5f} -> {rel_verdict}")
        irr_verdict = 'EQUIVALENT to chance' if irr_vs_chance['equivalent'] else 'not equivalent to chance'
        print(f"irrelevant vs chance (0.5): t={irr_vs_chance['t']:.3f}, p={irr_vs_chance['p']:.5f}, "
              f"TOST (bound=+/-0.05): p={irr_vs_chance['tost_p']:.5f} -> {irr_verdict}")

        means = [relevant.mean(), irrelevant.mean()]
        sems = [relevant.std(ddof=1) / np.sqrt(N_SEEDS), irrelevant.std(ddof=1) / np.sqrt(N_SEEDS)]
        ax.bar(['relevant', 'irrelevant'], means, yerr=sems, capsize=6, color=['crimson', 'purple'])
        ax.axhline(0.5, color='grey', linestyle=':', linewidth=1, label='chance')
        ax.set_ylim(0.4, 1.05)
        ax.set_ylabel('Cross-cue decoding accuracy')
        ax.set_title(f'{population} units (n={N_SEEDS})')
        ax.legend(loc='lower right', fontsize=8)

    fig.suptitle('gated_attractor: cross-cue decoding (train on one cue, test on the other)')
    fig.tight_layout()
    output_dir = os.path.join(_this_dir, 'gated_attractor', 'output')
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'cross_cue_decoding_n20.png')
    fig.savefig(fig_path, dpi=150)
    print(f'\nSaved {fig_path}')
