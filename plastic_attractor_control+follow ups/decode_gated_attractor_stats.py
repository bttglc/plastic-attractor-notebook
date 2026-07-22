"""Multi-seed, statistically tested version of decode_gated_attractor.py.

Each seed gets its own independent decode (own trials, own block-held-out
folds) -- seed is the unit of replication here, same convention as every
Control 1-3 script in this project. Each seed's relevant/irrelevant curves
are collapsed to one plateau-window mean each, using windows chosen from
the single-seed diagnostic run BEFORE any of these seeds were run (200-400
for conjunction units, which settle by ~step 200; 150-340 for feature
units, which are flat during the stimulus window but drop at ~step 350
offset) -- not picked after looking at this data, same a priori-margin
logic as the ±0.05 equivalence bound used in every other control here.

Run it from the project root:

    python decode_gated_attractor_stats.py
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import ttest_rel

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.getcwd()
sys.path.insert(0, _this_dir)
sys.path.insert(0, os.path.join(_this_dir, 'gated_attractor'))

from decode_gated_attractor import (
    decode_by_relevance, model_parameters, num_trials,
    practice_permutation_repeats, switch_probs,
)
from gated_attractor import SwitchingExperimentConfig, run_switching_experiment

N_SEEDS = 20
TIME_STEPS = np.arange(0, 400, 5)

# a priori plateau windows -- chosen from the single-seed diagnostic only,
# see module docstring
PLATEAU_WINDOWS = {
    'conjunction': (200, 400),
    'feature': (150, 340),
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


def run_one_seed(seed):
    """Returns the FULL curve per population, not just the plateau mean --
    kept so the full n=20 curves can be averaged into a time-course figure,
    not only collapsed into the scalar used for the paired t-test."""

    cfg = SwitchingExperimentConfig(
        seed=seed, num_trials=num_trials,
        practice_permutation_repeats=practice_permutation_repeats,
        switch_probs=switch_probs, model_parameters=model_parameters,
    )
    trials = run_switching_experiment(cfg).trials

    curves = {}
    for population in PLATEAU_WINDOWS:
        relevant, irrelevant = decode_by_relevance(
            trials, population=population, time_steps=TIME_STEPS,
        )
        curves[population] = (relevant, irrelevant)
    return curves


if __name__ == '__main__':
    print(f'Decoding across {N_SEEDS} seeds (2cpr_slowW3 preset)...')
    with ProcessPoolExecutor() as ex:
        results = []
        for i, r in enumerate(ex.map(run_one_seed, range(N_SEEDS)), start=1):
            results.append(r)
            print(f'  seed {i}/{N_SEEDS} done')

    print(f'\n=== Plateau-window decoding, n={N_SEEDS} seeds ===')
    fig_bar, axes_bar = plt.subplots(1, 2, figsize=(9, 4.5))
    fig_curve, axes_curve = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax_bar, ax_curve, population in zip(axes_bar, axes_curve, ('conjunction', 'feature')):
        win_start, win_stop = PLATEAU_WINDOWS[population]
        in_window = (TIME_STEPS >= win_start) & (TIME_STEPS < win_stop)

        # (n_seeds, n_timesteps) -- the full curves, not yet collapsed
        relevant_curves = np.stack([r[population][0] for r in results])
        irrelevant_curves = np.stack([r[population][1] for r in results])

        relevant = np.nanmean(relevant_curves[:, in_window], axis=1)
        irrelevant = np.nanmean(irrelevant_curves[:, in_window], axis=1)
        chance = np.full(N_SEEDS, 0.5)

        diff_stats = paired_comparison(relevant, irrelevant)
        chance_stats = paired_comparison(irrelevant, chance)

        print(f'\n--- {population} units (plateau window steps {win_start}-{win_stop}) ---')
        print(f'relevant:   {relevant.mean():.3f} (sd={relevant.std(ddof=1):.3f})')
        print(f'irrelevant: {irrelevant.mean():.3f} (sd={irrelevant.std(ddof=1):.3f})')
        print(f"relevant vs irrelevant: t={diff_stats['t']:.3f}, p={diff_stats['p']:.5f}, "
              f"diff={diff_stats['mean_diff']:+.3f} [{diff_stats['ci_low']:+.3f}, {diff_stats['ci_high']:+.3f}], "
              f"d={diff_stats['cohens_d']:.3f}")
        verdict = 'EQUIVALENT to chance' if chance_stats['equivalent'] else 'not established as equivalent to chance'
        print(f"irrelevant vs chance (0.5): t={chance_stats['t']:.3f}, p={chance_stats['p']:.5f}, "
              f"TOST (bound=+/-0.05): p={chance_stats['tost_p']:.5f} -> {verdict}")

        # bar chart: the plateau-window summary used for the stats test
        means = [relevant.mean(), irrelevant.mean()]
        sems = [relevant.std(ddof=1) / np.sqrt(N_SEEDS), irrelevant.std(ddof=1) / np.sqrt(N_SEEDS)]
        ax_bar.bar(['relevant', 'irrelevant'], means, yerr=sems, capsize=6,
                    color=['crimson', 'purple'])
        ax_bar.axhline(0.5, color='grey', linestyle=':', linewidth=1, label='chance')
        ax_bar.set_ylim(0.4, 1.05)
        ax_bar.set_ylabel('Plateau decoding accuracy')
        ax_bar.set_title(f'{population} units (n={N_SEEDS})')
        star = '*' if diff_stats['p'] < 0.05 else 'ns'
        ax_bar.text(0.5, 1.02, star, ha='center', transform=ax_bar.get_xaxis_transform())
        ax_bar.legend(loc='lower right', fontsize=8)

        # time-course: mean +/- SEM across all 20 seeds at every timestep --
        # the n=20 replacement for the old single-seed diagnostic plot
        rel_mean = np.nanmean(relevant_curves, axis=0)
        rel_sem = np.nanstd(relevant_curves, axis=0, ddof=1) / np.sqrt(N_SEEDS)
        irr_mean = np.nanmean(irrelevant_curves, axis=0)
        irr_sem = np.nanstd(irrelevant_curves, axis=0, ddof=1) / np.sqrt(N_SEEDS)

        ax_curve.plot(TIME_STEPS, rel_mean * 100, color='crimson', label='relevant')
        ax_curve.fill_between(TIME_STEPS, (rel_mean - rel_sem) * 100, (rel_mean + rel_sem) * 100,
                               color='crimson', alpha=0.25, linewidth=0)
        ax_curve.plot(TIME_STEPS, irr_mean * 100, color='purple', label='irrelevant')
        ax_curve.fill_between(TIME_STEPS, (irr_mean - irr_sem) * 100, (irr_mean + irr_sem) * 100,
                               color='purple', alpha=0.25, linewidth=0)
        ax_curve.axhline(50, color='grey', linestyle=':', linewidth=1)
        ax_curve.axvspan(win_start, win_stop, color='grey', alpha=0.08, label='plateau window')
        ax_curve.set_title(f'{population.capitalize()} units')
        ax_curve.set_xlabel('Time (timesteps)')
        ax_curve.set_ylim(45, 100)
        ax_curve.legend(fontsize=8)

    axes_curve[0].set_ylabel('Classification accuracy (%)')
    fig_bar.suptitle('gated_attractor: plateau-window decoding, relevant vs. irrelevant (n=20 seeds)')
    fig_bar.tight_layout()
    fig_curve.suptitle(f'gated_attractor (2cpr_slowW3): decoding over time, mean ± SEM across n={N_SEEDS} seeds')
    fig_curve.tight_layout()

    output_dir = os.path.join(_this_dir, 'gated_attractor', 'output')
    os.makedirs(output_dir, exist_ok=True)
    bar_path = os.path.join(output_dir, 'decoding_stats_n20.png')
    curve_path = os.path.join(output_dir, 'decoding_timecourse_n20.png')
    fig_bar.savefig(bar_path, dpi=150)
    fig_curve.savefig(curve_path, dpi=150)
    print(f'\nSaved {bar_path}')
    print(f'Saved {curve_path}')
