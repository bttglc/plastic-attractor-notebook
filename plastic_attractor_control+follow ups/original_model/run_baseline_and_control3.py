"""Run the published (original, cue-less) model's baseline plus the one
control that translates to its blocked, no-cue paradigm.

This model has no cue mechanism at all -- each block fully re-teaches its
rule via explicit instruction, and the rule *is* the block. Controls 1
(shuffle cue identity) and 2 (novel cue) have no structural equivalent here,
so only an adapted Control 3 is run: one (task, stimulus) combo is withheld
from every block's trials until a cutoff block, then let appear normally --
testing whether the network responds correctly to a stimulus combo it was
never shown, from the abstract feature-level rule alone (BlockedExperiment
Config.omit_combo / omit_combo_until_block, experiment.py).

Writes controls_summary.json in the same schema as every sibling model's
run_controls script (control1_*/control2_* left null here), so
compare_all_models.py can build one cross-model comparison.

Run it from the updated_model folder:

    python "original model/run_baseline_and_control3.py"
"""

import json
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
_project_root = (
    os.path.dirname(_this_dir)
    if os.path.basename(_this_dir) == 'original model'
    else _this_dir
)
sys.path.insert(0, _project_root)

# the folder name has a space, so it can't be `from "original model" import`
# -- load it directly from its path instead, same package either way.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    'original_model',
    os.path.join(_project_root, 'original model', '__init__.py'),
    submodule_search_locations=[os.path.join(_project_root, 'original model')],
)
original_model = importlib.util.module_from_spec(_spec)
sys.modules['original_model'] = original_model
_spec.loader.exec_module(original_model)

Task = original_model.Task
Feature = original_model.Feature
Stimulus = original_model.Stimulus
BlockedExperimentConfig = original_model.BlockedExperimentConfig
run_blocked_experiment = original_model.run_blocked_experiment
summarize_behavior = original_model.summarize_behavior
omitted_combo_contrast = original_model.omitted_combo_contrast
omitted_combo_exposure_sequence = original_model.omitted_combo_exposure_sequence

output_dir = os.path.join(_project_root, 'original model', 'output')

# RUN CONSTANTS #
# ===================================================== #

number_of_blocks = 20  # 10 COLOR + 10 SHAPE blocks, alternating
n_seeds = 20
n_exposure_bins = 5
# "practically negligible" bound for the TOST equivalence test below, in
# raw accuracy units (proportion correct). Default judgment call, not a
# validated literature value -- reconsider it for the specific claim made.
EQUIVALENCE_BOUND = 0.05

# omit (COLOR, green-square) from every COLOR block until block 10 (5 COLOR
# blocks' worth, since blocks alternate task), then let it appear normally
omit_combo = (Task.COLOR, Stimulus(Feature.GREEN, Feature.SQUARE))
omit_combo_until_block = 10

cpu_count = os.cpu_count() or 1


def build_configs(**overrides):
    return [
        BlockedExperimentConfig(seed=seed, number_of_blocks=number_of_blocks, **overrides)
        for seed in range(n_seeds)
    ]


def run_seeds_parallel(configs, label=''):
    with ProcessPoolExecutor(max_workers=min(cpu_count, n_seeds)) as executor:
        results = []
        for i, result in enumerate(executor.map(run_blocked_experiment, configs), start=1):
            print(f'  {label} seed {i}/{len(configs)} done')
            results.append(result)
        return results


def exposure_curve(sequences, n_bins=n_exposure_bins):
    bins = [[] for _ in range(n_bins)]
    for sequence in sequences:
        for i, trial in enumerate(sequence[:n_bins]):
            bins[i].append(float(trial.correct))
    means = [np.mean(b) if b else np.nan for b in bins]
    sters = [np.std(b) / np.sqrt(len(b)) if b else np.nan for b in bins]
    return means, sters


def sem(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values) / np.sqrt(len(values))) if len(values) else float('nan')


def paired_comparison(a, b, equivalence_bound=EQUIVALENCE_BOUND, alpha=0.05):
    """Full paired comparison between a and b: standard paired t-test, the
    95% CI of the mean difference, Cohen's d (paired, using the SD of the
    differences), and a TOST equivalence test against +/-equivalence_bound.

    A significant t-test argues the two conditions differ. A large p-value
    from that same test does NOT argue the reverse -- failing to reject
    "these differ" isn't evidence for "these are equivalent." TOST is the
    test that actually licenses an equivalence claim: it runs two one-sided
    tests (is the true difference above -bound? below +bound?) and only
    concludes equivalence if both null hypotheses of "a real difference at
    least this large" are rejected. tost_p here is the larger (weaker) of
    the two one-sided p-values, i.e. the standard conservative TOST p-value.
    """

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
        p_upper = scipy_stats.t.cdf(t_upper, df=n - 1)          # H0: true diff >= +bound
        t_lower = (mean_diff + equivalence_bound) / se_diff
        p_lower = 1 - scipy_stats.t.cdf(t_lower, df=n - 1)      # H0: true diff <= -bound
        tost_p = max(p_lower, p_upper)
    else:
        tost_p = 0.0 if abs(mean_diff) < equivalence_bound else 1.0
    equivalent = tost_p < alpha

    return {
        'mean_diff': mean_diff, 'ci_low': float(ci_low), 'ci_high': float(ci_high),
        'cohens_d': float(cohens_d), 't': float(t_stat), 'p': float(p_value),
        'tost_p': float(tost_p), 'equivalent': bool(equivalent),
        'equivalence_bound': equivalence_bound,
    }


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)

    print('Running baseline...')
    baseline_results = run_seeds_parallel(build_configs(), label='baseline')

    print('Running adapted Control 3 (omitted combo, withheld until block '
          f'{omit_combo_until_block})...')
    control3_results = run_seeds_parallel(
        build_configs(omit_combo=omit_combo, omit_combo_until_block=omit_combo_until_block),
        label='control3',
    )
    print('All simulations complete.\n')

    # ---------------- ACCURACY + SIGNIFICANCE ----------------
    seed_acc_baseline = np.array([summarize_behavior(r.trials).accuracy for r in baseline_results])
    seed_acc_c3 = np.array([summarize_behavior(r.trials).accuracy for r in control3_results])

    c3_contrasts = [omitted_combo_contrast(r) for r in control3_results]
    other_acc_c3 = np.array([c.easy_accuracy for c in c3_contrasts])
    omitted_acc_c3 = np.array([c.hard_accuracy for c in c3_contrasts])
    # this control asks whether omitted performs EQUIVALENTLY to other, not
    # just whether the standard t-test fails to find a difference -- see
    # paired_comparison's docstring and the tost_p / equivalent fields below.
    stats_omit = paired_comparison(omitted_acc_c3, other_acc_c3)
    t_omit, p_omit = stats_omit['t'], stats_omit['p']

    # ---------------- EXPOSURE CURVE ----------------
    omitted_sequences = [omitted_combo_exposure_sequence(r) for r in control3_results]
    omitted_exposure_mean, omitted_exposure_ster = exposure_curve(omitted_sequences)

    # ---------------- PLOTS ----------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    labels = ['Baseline', 'Control 3\n(omitted combo run)']
    means = [np.mean(seed_acc_baseline), np.mean(seed_acc_c3)]
    stds = [np.std(seed_acc_baseline) / np.sqrt(n_seeds), np.std(seed_acc_c3) / np.sqrt(n_seeds)]
    ax[0].bar(labels, means, yerr=stds, color=['blue', 'orange'], alpha=.7)
    ax[0].set_ylim(0, 1.1)
    ax[0].set_title('Overall accuracy (original model)\nControls 1/2 not applicable: no cue mechanism')

    labels3 = ['Other\ncombos', 'Omitted\ncombo (all)']
    means3 = [np.mean(other_acc_c3), np.mean(omitted_acc_c3)]
    stds3 = [np.std(other_acc_c3) / np.sqrt(n_seeds), np.std(omitted_acc_c3) / np.sqrt(n_seeds)]
    ax[1].bar(labels3, means3, yerr=stds3, color=['navajowhite', 'darkorange'])
    ax[1].set_ylim(0, 1.1)
    eq_label = 'equiv.' if stats_omit['equivalent'] else 'not equiv.'
    ax[1].set_title(
        f'Control 3: other vs omitted combo\nt={t_omit:.2f}, p={p_omit:.4f}'
        f"{' *' if p_omit < .05 else ''}\n"
        f"diff={stats_omit['mean_diff']:+.3f} [{stats_omit['ci_low']:+.3f}, {stats_omit['ci_high']:+.3f}], "
        f"d={stats_omit['cohens_d']:.2f}\n"
        f"TOST (±{stats_omit['equivalence_bound']:.2f}): p={stats_omit['tost_p']:.4f} ({eq_label})"
    )

    exposure_x = np.arange(1, n_exposure_bins + 1)
    ax[2].errorbar(exposure_x, omitted_exposure_mean, yerr=omitted_exposure_ster, color='orange', marker='o')
    ax[2].axhline(np.mean(other_acc_c3), color='darkorange', linestyle='--', label='Other-combo mean')
    ax[2].set_xticks(exposure_x); ax[2].set_ylim(0, 1.1)
    ax[2].set_xlabel('Omitted-combo exposure #'); ax[2].set_ylabel('Accuracy')
    ax[2].set_title('Omitted-combo learning curve'); ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'controls_comparison.png'), dpi=300)
    plt.show()

    # ---------------- SUMMARY ----------------
    print('\n=== SUMMARY ===')
    print(f'Baseline accuracy:              {np.mean(seed_acc_baseline):.3f}')
    print(f'Control 3 run overall accuracy: {np.mean(seed_acc_c3):.3f}')
    print('Controls 1 (shuffled cue) and 2 (novel cue): not applicable -- this model has no cue mechanism.')
    print()
    print(f'Control 3 - other combos:        {np.mean(other_acc_c3):.3f}')
    print(f'Control 3 - omitted combo (all): {np.mean(omitted_acc_c3):.3f}')
    print(f'  -> paired t-test omitted vs other: t={t_omit:.3f}, p={p_omit:.5f}')
    print(f"  -> mean diff (95% CI): {stats_omit['mean_diff']:+.3f} "
          f"[{stats_omit['ci_low']:+.3f}, {stats_omit['ci_high']:+.3f}], Cohen's d={stats_omit['cohens_d']:.3f}")
    print(f"  -> TOST equivalence test (bound=+/-{stats_omit['equivalence_bound']:.2f}): "
          f"p={stats_omit['tost_p']:.5f} -> "
          f"{'EQUIVALENT' if stats_omit['equivalent'] else 'not established as equivalent'}")
    print(f'  -> accuracy by omitted-combo exposure # (1-{n_exposure_bins}): '
          f'{[round(float(v), 3) for v in omitted_exposure_mean]}')

    print('Done: all simulations and plots complete.')

    # ---------------- JSON SUMMARY EXPORT (for compare_all_models.py) ----------------
    summary = {
        'model': 'original_model',
        'n_seeds': n_seeds,
        'equivalence_bound': EQUIVALENCE_BOUND,
        'baseline_accuracy_mean': float(np.mean(seed_acc_baseline)),
        'baseline_accuracy_sem': sem(seed_acc_baseline),
        'control1_accuracy_mean': None, 'control1_accuracy_sem': None,
        'control1_t': None, 'control1_p': None,
        'control1_ci': None, 'control1_cohens_d': None,
        'control1_tost_p': None, 'control1_equivalent': None,
        'control2_trained_mean': None, 'control2_trained_sem': None,
        'control2_novel_mean': None, 'control2_novel_sem': None,
        'control2_t': None, 'control2_p': None,
        'control2_ci': None, 'control2_cohens_d': None,
        'control2_tost_p': None, 'control2_equivalent': None,
        'control3_other_mean': float(np.mean(other_acc_c3)),
        'control3_other_sem': sem(other_acc_c3),
        'control3_omitted_mean': float(np.mean(omitted_acc_c3)),
        'control3_omitted_sem': sem(omitted_acc_c3),
        'control3_t': stats_omit['t'], 'control3_p': stats_omit['p'],
        'control3_ci': [stats_omit['ci_low'], stats_omit['ci_high']],
        'control3_cohens_d': stats_omit['cohens_d'],
        'control3_tost_p': stats_omit['tost_p'], 'control3_equivalent': stats_omit['equivalent'],
    }
    with open(os.path.join(output_dir, 'controls_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {os.path.join(output_dir, 'controls_summary.json')}")
