"""Run baseline plus the three experimental controls and plot the comparison.

Companion to launcher.py, focused on Controls 1-3 (see
SwitchingExperimentConfig.shuffle_cues_test / practice_cue_restriction /
omit_practice_combo) rather than model-version comparison. Same design as
cued_attractor/run_controls.py, applied to the gated model: restricting or
omitting a cue during practice/performance-practice also keeps the gate's
own cue-input weights from forming a genuine, rule-predictive association
for it (see experiment.py's _balanced_shuffled_deck docstring) -- verified
by direct inspection, not just assumed.

Also writes a small controls_summary.json next to the figure, in the same
schema used by every sibling model's run_controls script, so
compare_all_models.py can build one cross-model comparison without needing
to understand each model's internal trial format.

Run it from the updated_model folder:

    python gated_attractor/run_controls.py
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import ttest_rel

# make the gated_attractor package (this file's parent dir) importable
# regardless of the cwd, and handle running inside a Jupyter/Colab cell
# (no __file__) the same way as every sibling run_controls.py
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.getcwd()
_project_root = (
    os.path.dirname(_this_dir)
    if os.path.basename(_this_dir) == 'gated_attractor'
    else _this_dir
)
sys.path.insert(0, _project_root)

from gated_attractor import (
    SwitchingExperimentConfig,
    Task,
    build_vocabulary,
    cues_for_task,
    novel_cue_contrast,
    novel_cue_exposure_sequence,
    omitted_combo_contrast,
    omitted_combo_exposure_sequence,
    run_switching_experiment,
    summarize_behavior,
)
from gated_attractor.task import ALL_STIMULI
from model_versions_config import model_versions

output_dir = os.path.join(_project_root, 'gated_attractor', 'output', 'controls')

# RUN CONSTANTS #
# ===================================================== #

num_trials = 48
practice_permutation_repeats = 5
switch_probs = (0.125, 0.25, 0.5, 0.75) * 2
n_seeds = 20
n_exposure_bins = 5
# "practically negligible" bound for the TOST equivalence tests below, in
# raw accuracy units (proportion correct). This is a default judgment call,
# not a validated literature value -- reconsider it for the specific claim
# being made (e.g. a stricter bound like 0.03 for a stronger equivalence
# claim, a looser one like 0.08 if 5 points of accuracy genuinely wouldn't
# matter for the argument being made).
EQUIVALENCE_BOUND = 0.05
# '2cpr_gating_units' is the base gating-on preset, but its fast-dominant
# weights leave real-block accuracy at chance (~0.53, verified directly --
# routing instability documented in model_outline.md secs 13-14). 'slowW3'
# inverts which weight timescale dominates and is model_outline.md's own
# validated fix (0.565 -> 0.881 real-block accuracy at fast-dominant vs
# slow-dominant W); confirmed here too (n=6: mean 0.846). Controls run
# against an at-chance baseline wouldn't mean anything, so this is the
# preset that actually lets Controls 1-3 test something.
model_parameters = model_versions['2cpr_slowW3']

vocabulary = build_vocabulary(model_parameters.num_cues_per_rule)
color_cues = cues_for_task(vocabulary, Task.COLOR)
shape_cues = cues_for_task(vocabulary, Task.SHAPE)

# Control 2: teach only the first cue of each rule; the other cue stays
# novel until the real (test) blocks.
practice_cue_restriction = {Task.COLOR: (color_cues[0],), Task.SHAPE: (shape_cues[0],)}
# Control 3: omit one (cue, stimulus) pairing from every practice /
# performance-practice block.
omit_practice_combo = (color_cues[0], ALL_STIMULI[0])

cpu_count = os.cpu_count() or 1


def build_configs(**overrides):
    return [
        SwitchingExperimentConfig(
            seed=seed, num_trials=num_trials, practice_permutation_repeats=practice_permutation_repeats,
            switch_probs=switch_probs, model_parameters=model_parameters, **overrides,
        )
        for seed in range(n_seeds)
    ]


def run_seeds_parallel(configs, label=''):
    with ProcessPoolExecutor(max_workers=min(cpu_count, n_seeds)) as executor:
        results = []
        for i, result in enumerate(executor.map(run_switching_experiment, configs), start=1):
            print(f'  {label} seed {i}/{len(configs)} done')
            results.append(result)
        return results


def real_accuracy(result):
    """Accuracy over real (test) trials only, excluding practice/performance
    practice."""

    real_trials = [t for t in result.trials if not t.is_practice]
    return summarize_behavior(real_trials).accuracy


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


def exposure_curve(sequences, n_bins=n_exposure_bins):
    """Mean +/- SEM accuracy at each exposure number (1st, 2nd, ...)."""

    bins = [[] for _ in range(n_bins)]
    for sequence in sequences:
        for i, trial in enumerate(sequence[:n_bins]):
            bins[i].append(float(trial.correct))
    means = [np.mean(b) if b else np.nan for b in bins]
    sters = [np.std(b) / np.sqrt(len(b)) if b else np.nan for b in bins]
    return means, sters


def first_exposure_trial_index(sequence, config):
    """Trial number within the real/test phase (0 = first trial of the first
    real block) of the first trial in sequence."""

    performance_practice_blocks = 2 if config.include_performance_practice else 0
    real_blocks_start = config.num_practice_blocks + performance_practice_blocks
    first = sequence[0]
    return (first.block_index - real_blocks_start) * num_trials + first.trial_index_in_block


def sem(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values) / np.sqrt(len(values))) if len(values) else float('nan')


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)

    print('Running baseline...')
    baseline_results = run_seeds_parallel(build_configs(), label='baseline')

    print('Running Control 1 (shuffled cues at test)...')
    control1_results = run_seeds_parallel(build_configs(shuffle_cues_test=True), label='control1')

    print('Running Control 2 (novel cue generalization)...')
    control2_results = run_seeds_parallel(
        build_configs(practice_cue_restriction=practice_cue_restriction), label='control2'
    )

    print('Running Control 3 (omitted combo)...')
    control3_results = run_seeds_parallel(
        build_configs(omit_practice_combo=omit_practice_combo), label='control3'
    )
    print('All simulations complete.\n')

    # ---------------- ACCURACY + SIGNIFICANCE ----------------
    seed_acc_baseline = np.array([real_accuracy(r) for r in baseline_results])
    seed_acc_c1 = np.array([real_accuracy(r) for r in control1_results])
    seed_acc_c2 = np.array([real_accuracy(r) for r in control2_results])
    seed_acc_c3 = np.array([real_accuracy(r) for r in control3_results])

    # Control 1 asks whether these DIFFER (shuffling should hurt); Controls
    # 2/3 ask the opposite -- whether novel/omitted performs EQUIVALENTLY to
    # trained/other -- which is what the TOST fields in each stats dict are
    # for (see paired_comparison's docstring).
    stats_shuffle = paired_comparison(seed_acc_baseline, seed_acc_c1)
    t_shuffle, p_shuffle = stats_shuffle['t'], stats_shuffle['p']

    c2_contrasts = [novel_cue_contrast(r) for r in control2_results]
    trained_acc_c2 = np.array([c.easy_accuracy for c in c2_contrasts])
    novel_acc_c2 = np.array([c.hard_accuracy for c in c2_contrasts])
    stats_novel = paired_comparison(trained_acc_c2, novel_acc_c2)
    t_novel, p_novel = stats_novel['t'], stats_novel['p']

    c3_contrasts = [omitted_combo_contrast(r) for r in control3_results]
    other_acc_c3 = np.array([c.easy_accuracy for c in c3_contrasts])
    omitted_acc_c3 = np.array([c.hard_accuracy for c in c3_contrasts])
    stats_omit = paired_comparison(omitted_acc_c3, other_acc_c3)
    t_omit, p_omit = stats_omit['t'], stats_omit['p']

    # ---------------- EXPOSURE CURVES ----------------
    novel_sequences = [novel_cue_exposure_sequence(r) for r in control2_results]
    novel_exposure_mean, novel_exposure_ster = exposure_curve(novel_sequences)
    first_novel_trial = [
        first_exposure_trial_index(seq, r.config) for seq, r in zip(novel_sequences, control2_results) if seq
    ]

    omitted_sequences = [omitted_combo_exposure_sequence(r) for r in control3_results]
    omitted_exposure_mean, omitted_exposure_ster = exposure_curve(omitted_sequences)
    first_omitted_trial = [
        first_exposure_trial_index(seq, r.config) for seq, r in zip(omitted_sequences, control3_results) if seq
    ]

    # ---------------- PLOTS ----------------
    fig, ax = plt.subplots(2, 4, figsize=(24, 10))

    labels = ['Baseline', 'Shuffled\ncues', 'Novel-cue\nrestriction', 'Omitted\ncombo']
    means = [np.mean(a) for a in (seed_acc_baseline, seed_acc_c1, seed_acc_c2, seed_acc_c3)]
    stds = [np.std(a) / np.sqrt(n_seeds) for a in (seed_acc_baseline, seed_acc_c1, seed_acc_c2, seed_acc_c3)]
    ax[0, 0].bar(labels, means, yerr=stds, color=['blue', 'red', 'green', 'orange'], alpha=.7)
    ax[0, 0].set_ylim(0, 1.1); ax[0, 0].set_title('Real-block accuracy by condition (gated_attractor)')

    ax[0, 1].axis('off')

    def _stats_block(title, s):
        eq = 'equiv.' if s['equivalent'] else 'not equiv.'
        return (
            f"{title}:\n"
            f"  t={s['t']:.2f}, p={s['p']:.4f}{' *' if s['p'] < .05 else ''}\n"
            f"  diff={s['mean_diff']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}], d={s['cohens_d']:.2f}\n"
            f"  TOST (±{s['equivalence_bound']:.2f}): p={s['tost_p']:.4f} ({eq})"
        )

    stats_text = (
        "Paired t-tests + TOST equivalence (df = n_seeds-1)\n\n"
        + _stats_block('Baseline vs shuffled cues', stats_shuffle) + "\n\n"
        + _stats_block('Trained vs novel cue (C2)', stats_novel) + "\n\n"
        + _stats_block('Omitted vs other combo (C3)', stats_omit) + "\n\n"
        "* p < .05.  TOST: equiv. means the difference is significantly\n"
        "smaller than the ±bound -- a real equivalence claim, not just\n"
        "a failure to reject the standard t-test."
    )
    ax[0, 1].text(0.02, 0.98, stats_text, transform=ax[0, 1].transAxes,
                  va='top', ha='left', fontsize=8.5, family='monospace')

    labels2 = ['Trained\ncues', 'Novel\ncues (all)']
    means2 = [np.mean(trained_acc_c2), np.mean(novel_acc_c2)]
    stds2 = [np.std(trained_acc_c2) / np.sqrt(n_seeds), np.std(novel_acc_c2) / np.sqrt(n_seeds)]
    ax[0, 2].bar(labels2, means2, yerr=stds2, color=['seagreen', 'lightgreen'])
    ax[0, 2].set_ylim(0, 1.1); ax[0, 2].set_title('Control 2: trained vs novel cue')

    labels3 = ['Other\ncombos', 'Omitted\ncombo (all)']
    means3 = [np.mean(other_acc_c3), np.mean(omitted_acc_c3)]
    stds3 = [np.std(other_acc_c3) / np.sqrt(n_seeds), np.std(omitted_acc_c3) / np.sqrt(n_seeds)]
    ax[0, 3].bar(labels3, means3, yerr=stds3, color=['navajowhite', 'darkorange'])
    ax[0, 3].set_ylim(0, 1.1); ax[0, 3].set_title('Control 3: other vs omitted combo')

    exposure_x = np.arange(1, n_exposure_bins + 1)
    ax[1, 0].errorbar(exposure_x, novel_exposure_mean, yerr=novel_exposure_ster, color='green', marker='o')
    ax[1, 0].axhline(np.mean(trained_acc_c2), color='seagreen', linestyle='--', label='Trained-cue mean')
    ax[1, 0].set_xticks(exposure_x); ax[1, 0].set_ylim(0, 1.1)
    ax[1, 0].set_xlabel('Novel-cue exposure #'); ax[1, 0].set_ylabel('Accuracy')
    ax[1, 0].set_title('Control 2: novel-cue learning curve'); ax[1, 0].legend()

    ax[1, 1].errorbar(exposure_x, omitted_exposure_mean, yerr=omitted_exposure_ster, color='orange', marker='o')
    ax[1, 1].axhline(np.mean(other_acc_c3), color='darkorange', linestyle='--', label='Other-combo mean')
    ax[1, 1].set_xticks(exposure_x); ax[1, 1].set_ylim(0, 1.1)
    ax[1, 1].set_xlabel('Omitted-combo exposure #'); ax[1, 1].set_ylabel('Accuracy')
    ax[1, 1].set_title('Control 3: omitted-combo learning curve'); ax[1, 1].legend()

    ax[1, 2].hist(first_novel_trial, bins=10, color='green', alpha=.7)
    ax[1, 2].set_xlabel('Trial # of first novel-cue exposure')
    ax[1, 2].set_ylabel('Seeds'); ax[1, 2].set_title('Control 2: when novel cue first appears')

    ax[1, 3].hist(first_omitted_trial, bins=10, color='orange', alpha=.7)
    ax[1, 3].set_xlabel('Trial # of first omitted-combo exposure')
    ax[1, 3].set_ylabel('Seeds'); ax[1, 3].set_title('Control 3: when omitted combo first appears')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'controls_comparison.png'), dpi=300)
    plt.show()

    # ---------------- SUMMARY ----------------
    def _print_stats(label, s):
        eq = 'EQUIVALENT' if s['equivalent'] else 'not established as equivalent'
        print(f'  -> paired t-test {label}: t={s["t"]:.3f}, p={s["p"]:.5f}')
        print(f'  -> mean diff (95% CI): {s["mean_diff"]:+.3f} [{s["ci_low"]:+.3f}, {s["ci_high"]:+.3f}], '
              f"Cohen's d={s['cohens_d']:.3f}")
        print(f"  -> TOST equivalence test (bound=+/-{s['equivalence_bound']:.2f}): "
              f"p={s['tost_p']:.5f} -> {eq}")

    print('\n=== SUMMARY ===')
    print(f'Baseline accuracy:              {np.mean(seed_acc_baseline):.3f}')
    print(f'Shuffled-cue accuracy:          {np.mean(seed_acc_c1):.3f}')
    _print_stats('baseline vs shuffled', stats_shuffle)
    print()
    print(f'Control 2 - trained cues:        {np.mean(trained_acc_c2):.3f}')
    print(f'Control 2 - novel cues (all):    {np.mean(novel_acc_c2):.3f}')
    _print_stats('trained vs novel', stats_novel)
    print(f'  -> mean trial of first novel-cue exposure: {np.mean(first_novel_trial):.1f}')
    print(f'  -> accuracy by novel-cue exposure # (1-{n_exposure_bins}): '
          f'{[round(float(v), 3) for v in novel_exposure_mean]}')
    print()
    print(f'Control 3 - other combos:        {np.mean(other_acc_c3):.3f}')
    print(f'Control 3 - omitted combo (all): {np.mean(omitted_acc_c3):.3f}')
    _print_stats('omitted vs other', stats_omit)
    print(f'  -> mean trial of first omitted-combo exposure: {np.mean(first_omitted_trial):.1f}')
    print(f'  -> accuracy by omitted-combo exposure # (1-{n_exposure_bins}): '
          f'{[round(float(v), 3) for v in omitted_exposure_mean]}')

    print('\nNote: TOST equivalence is only a meaningful claim for Controls 2/3 '
          '("performs just as well as"); Control 1 is testing for a difference, not equivalence.')
    print('Done: all simulations and plots complete.')

    # ---------------- JSON SUMMARY EXPORT (for compare_all_models.py) ----------------
    summary = {
        'model': 'gated_attractor',
        'n_seeds': n_seeds,
        'equivalence_bound': EQUIVALENCE_BOUND,
        'baseline_accuracy_mean': float(np.mean(seed_acc_baseline)),
        'baseline_accuracy_sem': sem(seed_acc_baseline),
        'control1_accuracy_mean': float(np.mean(seed_acc_c1)),
        'control1_accuracy_sem': sem(seed_acc_c1),
        'control1_t': stats_shuffle['t'], 'control1_p': stats_shuffle['p'],
        'control1_ci': [stats_shuffle['ci_low'], stats_shuffle['ci_high']],
        'control1_cohens_d': stats_shuffle['cohens_d'],
        'control1_tost_p': stats_shuffle['tost_p'], 'control1_equivalent': stats_shuffle['equivalent'],
        'control2_trained_mean': float(np.mean(trained_acc_c2)),
        'control2_trained_sem': sem(trained_acc_c2),
        'control2_novel_mean': float(np.mean(novel_acc_c2)),
        'control2_novel_sem': sem(novel_acc_c2),
        'control2_t': stats_novel['t'], 'control2_p': stats_novel['p'],
        'control2_ci': [stats_novel['ci_low'], stats_novel['ci_high']],
        'control2_cohens_d': stats_novel['cohens_d'],
        'control2_tost_p': stats_novel['tost_p'], 'control2_equivalent': stats_novel['equivalent'],
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
