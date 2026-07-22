"""Control 3, congruency-matched: does the omitted combo suffer beyond what
incongruent trials cost in general?

Two earlier versions of this control both had a confound:

1. The original version always omitted (cue A1, green-square) -- the
   CONGRUENT stimulus, in every model's own is_congruent() definition. Every
   model showed no deficit (sometimes even a numeric advantage) for the
   omitted combo -- but that's expected if the omitted case is simply the
   easy stimulus type, independent of whether it was ever trained.

2. Swapping to an incongruent omitted combo (cue A1, green-circle) reversed
   this completely: all four models showed a significant deficit. But that
   comparison is against "all other combos", a mixed pool of congruent AND
   incongruent trials -- so part of that deficit could just be "incongruent
   trials are harder in general", not "this specific never-trained pairing
   is harder than trained incongruent pairings".

This script is the properly isolated version: it restricts the comparison
to INCONGRUENT trials only, comparing the omitted incongruent combo against
OTHER, trained incongruent combos (different cue and/or different
incongruent stimulus). Congruency effects are a documented, expected
property of this whole task design -- this comparison holds that effect
constant so that what's left is specifically about whether the network
generalizes an untrained pairing, not about congruency at all.

Run it yourself:

    python3 congruency_matched_control3.py

Runs all four models at n=20 seeds (same scale as every other controls
script in this project). Each model runs sequentially, not in parallel,
since each one already parallelizes internally across every CPU core you
have -- running them concurrently would just make them compete.
"""

import sys
import os
import importlib.util
import json

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import stats as scipy_stats
from scipy.stats import ttest_rel

try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT = os.getcwd()
sys.path.insert(0, PROJECT_ROOT)

N_SEEDS = 20
EQUIVALENCE_BOUND = 0.05
output_dir = os.path.join(PROJECT_ROOT, 'output')


# ---------------- shared stats helper (same as every other script here) ----------------

def paired_comparison(a, b, equivalence_bound=EQUIVALENCE_BOUND, alpha=0.05):
    """Paired t-test + 95% CI of the mean difference + Cohen's d + a TOST
    equivalence test against +/-equivalence_bound. See any other
    run_controls.py in this project for the full rationale."""

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
    return {
        'mean_diff': mean_diff, 'ci_low': float(ci_low), 'ci_high': float(ci_high),
        'cohens_d': float(cohens_d), 't': float(t_stat), 'p': float(p_value),
        'tost_p': float(tost_p), 'equivalent': bool(equivalent),
    }


def report(name, other_incongruent, omitted):
    other_incongruent = np.array(other_incongruent, dtype=float)
    omitted = np.array(omitted, dtype=float)
    s = paired_comparison(omitted, other_incongruent)
    verdict = 'EQUIVALENT' if s['equivalent'] else 'not equivalent'
    print(f'\n=== {name}: congruency-matched Control 3 (n={N_SEEDS}) ===')
    print(f'other incongruent combos: {np.mean(other_incongruent):.3f}')
    print(f'omitted (incongruent):    {np.mean(omitted):.3f}')
    print(f"t={s['t']:.3f}, p={s['p']:.5f}, diff={s['mean_diff']:+.3f} "
          f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}], d={s['cohens_d']:.3f}")
    print(f"TOST (bound=+/-{EQUIVALENCE_BOUND:.2f}): p={s['tost_p']:.5f} -> {verdict}")
    return s


def load_module(name, path, pkg_dir=None):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[pkg_dir] if pkg_dir else None
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_run_controls(unique_name, pkg_dir):
    """cued_attractor/run_controls.py and gated_attractor/run_controls.py both
    do a bare `from model_versions_config import model_versions` -- with both
    package directories on sys.path at once, the second import would reuse
    whichever bare `model_versions_config` got cached first (same generic
    name, different file per package), silently grabbing the WRONG preset
    dict. Force a clean re-resolution before each package load."""

    sys.modules.pop('model_versions_config', None)
    sys.path[:] = [
        p for p in sys.path
        if not p.endswith('/cued_attractor') and not p.endswith('/gated_attractor')
    ]
    sys.path.insert(0, pkg_dir)
    return load_module(unique_name, f"{pkg_dir}/run_controls.py", pkg_dir)


# incongruent stimulus indices, for the two OOP packages (ALL_STIMULI order:
# green-square[congruent], green-circle[incongruent], blue-square[incongruent],
# blue-circle[congruent])
INCONGRUENT_STIMULUS_INDICES = (1, 2)


# ---------------- cued_attractor ----------------
ca_rc = load_run_controls("ca_run_controls", f"{PROJECT_ROOT}/cued_attractor")
ca_omit_combo = (ca_rc.color_cues[0], ca_rc.ALL_STIMULI[1])  # cue A1 + green-circle


def run_cued(seed):
    cfg = ca_rc.SwitchingExperimentConfig(
        seed=seed, num_trials=ca_rc.num_trials,
        practice_permutation_repeats=ca_rc.practice_permutation_repeats,
        switch_probs=ca_rc.switch_probs, model_parameters=ca_rc.model_parameters,
        omit_practice_combo=ca_omit_combo,
    )
    return ca_rc.run_switching_experiment(cfg)


def congruency_matched_split_oop(result, omit_combo, incongruent_stimuli):
    """For the two OOP packages (cued_attractor, gated_attractor): split real
    trials into (other incongruent combos, the omitted combo), both filtered
    to incongruent stimuli only."""

    real_trials = [t for t in result.trials if not t.is_practice]
    omitted = [t.correct for t in real_trials if (t.cue, t.stimulus) == omit_combo]
    other_incongruent = [
        t.correct for t in real_trials
        if t.stimulus in incongruent_stimuli and (t.cue, t.stimulus) != omit_combo
    ]
    return other_incongruent, omitted


# ---------------- gated_attractor ----------------
ga_rc = load_run_controls("ga_run_controls", f"{PROJECT_ROOT}/gated_attractor")
ga_omit_combo = (ga_rc.color_cues[0], ga_rc.ALL_STIMULI[1])


def run_gated(seed):
    cfg = ga_rc.SwitchingExperimentConfig(
        seed=seed, num_trials=ga_rc.num_trials,
        practice_permutation_repeats=ga_rc.practice_permutation_repeats,
        switch_probs=ga_rc.switch_probs, model_parameters=ga_rc.model_parameters,
        omit_practice_combo=ga_omit_combo,
    )
    return ga_rc.run_switching_experiment(cfg)


# ---------------- updated_model_v1 ----------------
v1 = load_module("attractor_v1", f"{PROJECT_ROOT}/attractor version1/attractor_control_analysis.py")
v1_switch_probs = (0.125, 0.25, 0.5, 0.75) * 2
v1_omit_combo = (0, 1)  # cue A1 (0) + stimulus 1 (green circle)
v1_incongruent_stim_indices = (1, 2)  # green-circle, blue-square


def run_v1(seed):
    res = v1.plasticattractor_sim(
        num_trials=48, rnd_seed=seed, switch_probs=v1_switch_probs, practice_trials=48,
        TMS_sim=False, TMS_start=None, shuffle_cues_test=False,
        practice_cue_restriction=None, omit_practice_combo=v1_omit_combo,
        **v1.model_versions['whyte_original'],
    )
    acc_dict, cue_lab_dict, stim_seq_dict = res[3], res[8], res[11]
    num_practice_blocks, num_trials = 2, 48
    num_real_blocks = len(v1_switch_probs)
    other_incongruent, omitted = [], []
    for blk_idx in range(num_real_blocks):
        actual_blk = blk_idx + num_practice_blocks
        acc = acc_dict[actual_blk]
        cues = np.array([cue_lab_dict[actual_blk][t] for t in range(len(acc))])
        stim = stim_seq_dict[actual_blk]
        is_incongruent = np.isin(stim, v1_incongruent_stim_indices)
        is_omitted = (cues == v1_omit_combo[0]) & (stim == v1_omit_combo[1])
        other_incongruent.extend(acc[is_incongruent & ~is_omitted])
        omitted.extend(acc[is_omitted])
    return (np.mean(other_incongruent) if other_incongruent else np.nan,
            np.mean(omitted) if omitted else np.nan)


# ---------------- original_model ----------------
om = load_module("original_model", f'{PROJECT_ROOT}/original model/__init__.py',
                  f'{PROJECT_ROOT}/original model')
om_omit_combo = (om.Task.COLOR, om.Stimulus(om.Feature.GREEN, om.Feature.CIRCLE))


def run_om(seed):
    cfg = om.BlockedExperimentConfig(
        seed=seed, number_of_blocks=20,
        omit_combo=om_omit_combo, omit_combo_until_block=10,
    )
    result = om.run_blocked_experiment(cfg)
    omitted = [
        t.correct for t in result.trials
        if t.task == om_omit_combo[0] and t.stimulus == om_omit_combo[1]
    ]
    other_incongruent = [
        t.correct for t in result.trials
        if not om.is_congruent(t.stimulus)
        and not (t.task == om_omit_combo[0] and t.stimulus == om_omit_combo[1])
    ]
    return (np.mean(other_incongruent) if other_incongruent else np.nan,
            np.mean(omitted) if omitted else np.nan)


if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)

    from cued_attractor.task import is_congruent
    print(f'Sanity check -- green-circle is_congruent: {is_congruent(ca_rc.ALL_STIMULI[1])} '
          f'(should be False)')
    print(f'Sanity check -- blue-square is_congruent:  {is_congruent(ca_rc.ALL_STIMULI[2])} '
          f'(should be False)')
    print('\nRunning congruency-matched Control 3 across all four models...')

    results = {}

    print('\ncued_attractor...')
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        ca_results = list(ex.map(run_cued, range(N_SEEDS)))
    other_ca, omit_ca = zip(*[
        congruency_matched_split_oop(r, ca_omit_combo, (ca_rc.ALL_STIMULI[1], ca_rc.ALL_STIMULI[2]))
        for r in ca_results
    ])
    other_ca = [np.mean(o) if o else np.nan for o in other_ca]
    omit_ca = [np.mean(o) if o else np.nan for o in omit_ca]
    results['cued_attractor'] = report('cued_attractor', other_ca, omit_ca)

    print('\ngated_attractor...')
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        ga_results = list(ex.map(run_gated, range(N_SEEDS)))
    other_ga, omit_ga = zip(*[
        congruency_matched_split_oop(r, ga_omit_combo, (ga_rc.ALL_STIMULI[1], ga_rc.ALL_STIMULI[2]))
        for r in ga_results
    ])
    other_ga = [np.mean(o) if o else np.nan for o in other_ga]
    omit_ga = [np.mean(o) if o else np.nan for o in omit_ga]
    results['gated_attractor'] = report('gated_attractor', other_ga, omit_ga)

    print('\nupdated_model_v1...')
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        v1_results = list(ex.map(run_v1, range(N_SEEDS)))
    results['updated_model_v1'] = report(
        'updated_model_v1', [r[0] for r in v1_results], [r[1] for r in v1_results]
    )

    print('\noriginal_model...')
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        om_results = list(ex.map(run_om, range(N_SEEDS)))
    results['original_model'] = report(
        'original_model', [r[0] for r in om_results], [r[1] for r in om_results]
    )

    with open(os.path.join(output_dir, 'congruency_matched_control3.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {os.path.join(output_dir, 'congruency_matched_control3.json')}")
    print('\nAll done.')
