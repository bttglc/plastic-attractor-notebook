"""
Plastic attractor network: task-switching simulation + experimental controls.

This consolidates the final, working iteration from untitled3.py (the earlier
two attempts in that file had bugs — an undefined cue_lab_dict / other_acc,
and a guessed-rather-than-actual stimulus sequence for Control 3 — both fixed
here by having the simulator return stim_seq_dict directly). Self-contained:
no import of a separate attractor_rnn module, and no Colab-only paths.

Controls:
  1. Shuffled cues at test — is the network actually reading the cue, or
     just pattern-matching on the stimulus?
  2. Novel-cue generalization — train on one cue per rule (A1, B1), test
     transfer to the untrained cue of the same rule (A2, B2).
  3. Omitted cue-stimulus combination — leave one (cue, stimulus) pairing
     out of practice entirely, test whether the network still gets it right.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.stats import ttest_rel


def plasticattractor_sim(num_trials, rnd_seed, switch_probs, practice_trials, TMS_sim, TMS_start,
                          alpha_cc_inhib=-.45, alpha_cc_decay=1, alpha_cf_gain=.08,
                          alpha_ff_inhib=-.28, alpha_ff_decay=.73, alpha_fc_gain=.04,
                          beta=.175, lr_short=.02, lr_long=.0002,
                          w_short_bound=1, w_long_bound=.2,
                          w_short_weight=1, w_long_weight=1,
                          shuffle_cues_test=False,
                          practice_cue_restriction=None,
                          omit_practice_combo=None):
    np.random.seed(rnd_seed)
    num_features = 10
    num_conjunction = 4
    alpha = np.array([alpha_cc_inhib, alpha_cc_decay, alpha_cf_gain,
                       alpha_ff_inhib, alpha_ff_decay, alpha_fc_gain])
    gamma = lr_short
    action_idx = slice(8, num_features)
    feature_groups = [[0, 1], [2, 3], [4, 5, 6, 7], [8, 9]]
    inhibition_mask = np.zeros([num_features, num_features])
    for group in feature_groups:
        inhibition_mask[np.ix_(group, group)] = 1
    W_ff = alpha[4] * np.eye(num_features) + alpha[3] * inhibition_mask
    W_cc = alpha[1] * np.eye(num_conjunction) + alpha[0] * np.ones(num_conjunction)

    cues = np.zeros([4, num_features])
    cues[0, 4] = 1   # cueA1
    cues[1, 5] = 1   # cueA2
    cues[2, 6] = 1   # cueB1
    cues[3, 7] = 1   # cueB2
    cue_rule = np.array([0, 0, 1, 1])

    stimuli = np.zeros([4, num_features])
    stimuli[0, :4] = [1, 0, 1, 0]   # green square
    stimuli[1, :4] = [1, 0, 0, 1]   # green circle
    stimuli[2, :4] = [0, 1, 1, 0]   # blue square
    stimuli[3, :4] = [0, 1, 0, 1]   # blue circle

    action_teach = np.array([[1, -1], [-1, 1]])
    num_practice_blocks = 2
    num_blocks = num_practice_blocks + len(switch_probs)
    stim_period = 400
    isi = np.full(num_features, -1)
    resp_period = np.zeros(num_features)

    def practice_sequence(rule_idx):
        cue_pool = np.where(cue_rule == rule_idx)[0]
        if practice_cue_restriction == 'A1_B1':
            if rule_idx == 0:
                cue_pool = [0]  # Only A1
            if rule_idx == 1:
                cue_pool = [2]  # Only B1
        combos = np.array([[cue, stim] for cue in cue_pool for stim in range(4)])
        if omit_practice_combo is not None:
            mask = ~((combos[:, 0] == omit_practice_combo[0]) &
                     (combos[:, 1] == omit_practice_combo[1]))
            combos = combos[mask]
        # Re-tile to match practice_trials approximately
        repeats = int(np.ceil(practice_trials / len(combos)))
        combos = np.tile(combos, (repeats, 1))[:practice_trials]
        np.random.shuffle(combos)
        return np.full(practice_trials, rule_idx), combos[:, 0], combos[:, 1]

    def real_sequence(switch_prob):
        rule_seq = np.zeros(num_trials, dtype=int)
        cue_seq = np.zeros(num_trials, dtype=int)
        rule_seq[0] = np.random.randint(2)
        for trl in range(1, num_trials):
            rule_seq[trl] = 1 - rule_seq[trl - 1] if np.random.rand() < switch_prob else rule_seq[trl - 1]
        for trl in range(num_trials):
            cue_seq[trl] = np.random.choice(np.where(cue_rule == rule_seq[trl])[0])
        if shuffle_cues_test:
            np.random.shuffle(cue_seq)
        stim_seq = np.tile([0, 1, 2, 3], num_trials // 4)
        np.random.shuffle(stim_seq)
        return rule_seq, cue_seq, stim_seq

    def stim_generator(rule_seq, cue_seq, stim_seq, practice):
        stimulus, ground_truth, c_labels, s_labels, r_labels, cue_labels, trans_labels = {}, {}, {}, {}, {}, {}, {}
        for trl in range(len(rule_seq)):
            r_idx, c_idx, s_idx = rule_seq[trl], cue_seq[trl], stim_seq[trl]
            correct = (0 if s_idx < 2 else 1) if r_idx == 0 else (0 if s_idx % 2 == 0 else 1)

            stim_input = np.zeros([num_features, stim_period])
            for t in range(stim_period):
                if t <= 50:
                    stim_input[:, t] = isi
                elif 50 <= t <= 100:
                    stim_input[:, t] = stimuli[s_idx, :] + cues[c_idx, :]
                elif 100 <= t <= 350:
                    stim_input[:, t] = resp_period
                elif 350 <= t <= 400:
                    stim_input[:, t] = isi
                if practice and 50 <= t <= 350:
                    stim_input[action_idx, t] = action_teach[correct, :]

            stimulus[trl], ground_truth[trl] = stim_input, correct
            r_labels[trl], cue_labels[trl] = r_idx, c_idx
            trans_labels[trl] = -1 if trl == 0 else (
                2 if rule_seq[trl] != rule_seq[trl - 1] else
                (1 if cue_seq[trl] != cue_seq[trl - 1] else 0)
            )
            c_labels[trl] = 1 if s_idx < 2 else 2
            s_labels[trl] = 1 if s_idx % 2 == 0 else 2
        return stimulus, ground_truth, c_labels, s_labels, r_labels, cue_labels, trans_labels

    conj_dict, feat_dict, choice_dict, acc_dict, rt_dict, weight_dict = {}, {}, {}, {}, {}, {}
    c_label_dict, s_label_dict, r_label_dict, cue_label_dict, trans_label_dict = {}, {}, {}, {}, {}
    # Kept alongside every block's trials so downstream analysis (novel-cue /
    # omitted-combo splits) doesn't have to guess the stimulus sequence.
    stim_seq_dict = {}
    # Raw rule sequence per block, so downstream analyses can cross novel-cue /
    # omitted-combo effects with switch-vs-repeat trials without decoding it
    # back out of trans_label_dict.
    rule_seq_dict = {}

    w_short, w_long = np.random.rand(num_features, num_conjunction), np.random.rand(num_features, num_conjunction)
    W = w_short_weight * w_short + w_long_weight * w_long

    for blk in range(num_blocks):
        practice = blk < num_practice_blocks
        if practice:
            rule_seq, cue_seq, stim_seq = practice_sequence(blk)
        else:
            rule_seq, cue_seq, stim_seq = real_sequence(switch_probs[blk - num_practice_blocks])
        n_trials = len(rule_seq)

        conjunction_units = np.zeros((num_conjunction, stim_period, n_trials))
        feature_units = np.zeros((num_features, stim_period, n_trials))
        choice, rt, accuracy = np.zeros(n_trials), np.zeros(n_trials), np.zeros(n_trials)

        stim_dict, g_truth, c_l, s_l, r_l, cue_l, t_l = stim_generator(rule_seq, cue_seq, stim_seq, practice)

        for trl in range(n_trials):
            stim = stim_dict[trl]
            for t in range(stim_period):
                feature_units[:, t, trl] = np.clip(
                    beta + W_ff @ (feature_units[:, t - 1, trl] - beta)
                    + alpha[2] * W @ (conjunction_units[:, t - 1, trl] - beta)
                    + stim[:, t], 0, 1)
                if TMS_sim and TMS_start <= t <= 100 and TMS_start != 100:
                    conjunction_units[:, t, trl] = 1
                else:
                    conjunction_units[:, t, trl] = np.clip(
                        beta + W_cc @ (conjunction_units[:, t - 1, trl] - beta)
                        + alpha[5] * W.T @ (feature_units[:, t - 1, trl] - beta)
                        + 0.005 * np.random.randn(num_conjunction), 0, 1)
                delta_w = np.outer((feature_units[:, t, trl] - beta), (conjunction_units[:, t, trl] - beta))
                w_short = np.clip(w_short + gamma * delta_w, 0, w_short_bound)
                w_long = np.clip(w_long + lr_long * delta_w, 0, w_long_bound)
                W = w_short_weight * w_short + w_long_weight * w_long
            max_act = np.amax(feature_units[action_idx, 110:, trl])
            act_idx = np.argwhere(feature_units[action_idx, 110:, trl] > 0.98 * max_act)
            choice[trl], rt[trl] = act_idx[0, 0], act_idx[0, 1]
            accuracy[trl] = 1 if choice[trl] == g_truth[trl] else 0
            weight_dict[trl, blk] = W

        conj_dict[blk], feat_dict[blk], choice_dict[blk] = conjunction_units, feature_units, choice
        acc_dict[blk], rt_dict[blk] = accuracy, rt
        c_label_dict[blk], s_label_dict[blk] = c_l, s_l
        r_label_dict[blk], cue_label_dict[blk], trans_label_dict[blk] = r_l, cue_l, t_l
        stim_seq_dict[blk] = np.asarray(stim_seq)
        rule_seq_dict[blk] = np.asarray(rule_seq)
        weight_dict[blk] = W

    return (conj_dict, feat_dict, choice_dict, acc_dict, rt_dict,
            c_label_dict, s_label_dict, r_label_dict, cue_label_dict, trans_label_dict,
            weight_dict, stim_seq_dict, rule_seq_dict)


# ------------------------------------------------------------------------- #
# CONFIG
# ------------------------------------------------------------------------- #

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ isn't defined when running inside a Jupyter/IPython/Colab cell
    base_dir = os.getcwd()
output_dir = os.path.join(base_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

n_seeds = 20
num_trials = 48
practice_trials = 48
switch_probs = [0.125, 0.25, 0.5, 0.75] * 2
num_practice_blocks = 2
num_real_blocks = len(switch_probs)
block_kinds = [0.125, 0.25, 0.5, 0.75]
switch_probs_arr = np.array(switch_probs)
# "practically negligible" bound for the TOST equivalence tests below, in
# raw accuracy units (proportion correct). Default judgment call, not a
# validated literature value -- reconsider it for the specific claim made.
EQUIVALENCE_BOUND = 0.05

model_versions = {
    'whyte_original': {
        'alpha_cc_inhib': -.45, 'alpha_cc_decay': 1, 'alpha_cf_gain': .08,
        'alpha_ff_inhib': -.28, 'alpha_ff_decay': .73, 'alpha_fc_gain': .04,
        'beta': .175, 'lr_short': .02, 'lr_long': .0002,
        'w_short_bound': 1, 'w_long_bound': .2,
        'w_short_weight': 1, 'w_long_weight': 1,
    },
}
active_version = 'whyte_original'
omit_combo = (0, 0)  # cue A1 (0) + stimulus 0 (green square)


def safe_mean(arr):
    return np.mean(arr) if np.size(arr) > 0 else np.nan


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


def run_sims(label, **kwargs):
    results = {}
    for rnd in range(n_seeds):
        results[rnd] = plasticattractor_sim(
            num_trials=num_trials, rnd_seed=rnd, switch_probs=switch_probs,
            practice_trials=practice_trials, TMS_sim=False, TMS_start=None,
            **model_versions[active_version], **kwargs)
        if (rnd + 1) % 5 == 0:
            print(f'{label} progress: {((rnd + 1) / n_seeds) * 100:.0f}%')
    return results


if __name__ == '__main__':
    print("Running baseline...")
    results_baseline = run_sims('Baseline', shuffle_cues_test=False,
                                 practice_cue_restriction=None, omit_practice_combo=None)

    print("Running Control 1 (shuffled cues)...")
    results_control1 = run_sims('Control 1', shuffle_cues_test=True,
                                 practice_cue_restriction=None, omit_practice_combo=None)

    print("Running Control 2 (novel cue generalization)...")
    results_control2 = run_sims('Control 2', shuffle_cues_test=False,
                                 practice_cue_restriction='A1_B1', omit_practice_combo=None)

    print("Running Control 3 (omitted combo)...")
    results_control3 = run_sims('Control 3', shuffle_cues_test=False,
                                 practice_cue_restriction=None, omit_practice_combo=omit_combo)

    print("All simulations complete.\n")

    # ---------------- EIGENVALUES + OVERALL ACCURACY ----------------
    def extract_eig_and_acc(results):
        sat_vals = np.zeros([n_seeds, num_trials, num_real_blocks])
        all_acc = []
        seed_acc = np.zeros(n_seeds)  # one accuracy value per seed, for paired stats
        for rnd in range(n_seeds):
            res = results[rnd]
            acc_dict, weights = res[3], res[10]
            block_accs = []
            for blk_idx in range(num_real_blocks):
                actual_blk = blk_idx + num_practice_blocks
                block_mean = np.mean(acc_dict[actual_blk])
                all_acc.append(block_mean)
                block_accs.append(block_mean)
                for trl in range(num_trials):
                    W = weights[trl, actual_blk]
                    vals = np.linalg.eigvals(W @ W.T)
                    sat_vals[rnd, trl, blk_idx] = np.sum(np.real(vals) > 1)
            seed_acc[rnd] = np.mean(block_accs)
        mean_sat = np.mean(np.mean(sat_vals, 0), 1)
        ster_sat = np.std(np.mean(sat_vals, 0), 1) / np.sqrt(n_seeds)
        return mean_sat, ster_sat, all_acc, seed_acc

    mean_sat_baseline, ster_sat_baseline, all_acc_baseline, seed_acc_baseline = extract_eig_and_acc(results_baseline)
    mean_sat_c1, ster_sat_c1, all_acc_c1, seed_acc_c1 = extract_eig_and_acc(results_control1)
    mean_sat_c2, ster_sat_c2, all_acc_c2, seed_acc_c2 = extract_eig_and_acc(results_control2)
    mean_sat_c3, ster_sat_c3, all_acc_c3, seed_acc_c3 = extract_eig_and_acc(results_control3)

    # ---------------- CONTROL 1: shuffle cost stratified by switch probability ----------------
    shuffle_cost_by_kind = np.zeros([n_seeds, len(block_kinds)])
    for rnd in range(n_seeds):
        accB, accS = results_baseline[rnd][3], results_control1[rnd][3]
        for k, kind in enumerate(block_kinds):
            kind_blocks = np.where(switch_probs_arr == kind)[0] + num_practice_blocks
            accB_kind = np.concatenate([accB[b] for b in kind_blocks])
            accS_kind = np.concatenate([accS[b] for b in kind_blocks])
            shuffle_cost_by_kind[rnd, k] = np.mean(accB_kind) - np.mean(accS_kind)
    mean_shuffle_cost = np.mean(shuffle_cost_by_kind, 0)
    ster_shuffle_cost = np.std(shuffle_cost_by_kind, 0) / np.sqrt(n_seeds)

    # ---------------- CONTROL 2: trained vs novel cue, exposure-by-exposure ----------------
    # The network keeps learning during testing (the Hebbian updates never turn
    # off), so pooling "all later novel-cue trials" together conflates trials
    # right after the novel cue's first appearance with trials where the network
    # has already adapted to it. Track exposure number explicitly instead of
    # just first-vs-later, and record when each novel cue first showed up.
    N_EXPOSURE_BINS = 5

    def first_exposure_analysis_c2(results, n_exposure_bins=N_EXPOSURE_BINS):
        first_exp_acc, later_novel_acc, trained_acc, novel_acc = [], [], [], []
        first_A2_trial, first_B2_trial = [], []
        exposure_curve = [[] for _ in range(n_exposure_bins)]
        for rnd in range(n_seeds):
            res = results[rnd]
            acc_dict, cue_lab_dict = res[3], res[8]
            novel_count = 0
            seen_A2, seen_B2 = False, False
            first_trial, later_trial, trained_trial, novel_trial = [], [], [], []
            exposure_bins = [[] for _ in range(n_exposure_bins)]
            for blk_idx in range(num_real_blocks):
                actual_blk = blk_idx + num_practice_blocks
                acc = acc_dict[actual_blk]
                cues = np.array([cue_lab_dict[actual_blk][trl] for trl in range(len(acc))])
                for trl in range(len(acc)):
                    c = cues[trl]
                    global_trial = blk_idx * num_trials + trl
                    if c in (1, 3):  # novel cues A2, B2 (never taught in practice)
                        novel_trial.append(acc[trl])
                        if novel_count == 0:
                            first_trial.append(acc[trl])
                        else:
                            later_trial.append(acc[trl])
                        if novel_count < n_exposure_bins:
                            exposure_bins[novel_count].append(acc[trl])
                        if c == 1 and not seen_A2:
                            first_A2_trial.append(global_trial)
                            seen_A2 = True
                        if c == 3 and not seen_B2:
                            first_B2_trial.append(global_trial)
                            seen_B2 = True
                        novel_count += 1
                    else:  # trained cues A1, B1
                        trained_trial.append(acc[trl])
            if first_trial: first_exp_acc.append(np.mean(first_trial))
            if later_trial: later_novel_acc.append(np.mean(later_trial))
            if trained_trial: trained_acc.append(np.mean(trained_trial))
            if novel_trial: novel_acc.append(np.mean(novel_trial))
            for i in range(n_exposure_bins):
                if exposure_bins[i]:
                    exposure_curve[i].append(np.mean(exposure_bins[i]))
        exposure_mean = [np.mean(e) if e else np.nan for e in exposure_curve]
        exposure_ster = [np.std(e) / np.sqrt(len(e)) if e else np.nan for e in exposure_curve]
        return {
            'first_exp_acc': first_exp_acc, 'later_acc': later_novel_acc,
            'trained_acc': trained_acc, 'novel_acc': novel_acc,
            'exposure_mean': exposure_mean, 'exposure_ster': exposure_ster,
            'first_A2_trial': first_A2_trial, 'first_B2_trial': first_B2_trial,
        }

    c2 = first_exposure_analysis_c2(results_control2)

    # ---------------- CONTROL 3: omitted combo, exposure-by-exposure ----------------
    # Same issue as Control 2, plus the omitted (cue, stimulus) pair can first
    # appear on trial 2 of testing for one seed and trial 46 for another, since
    # stim_seq is reshuffled per real block. Track the actual first-occurrence
    # trial and an exposure-number curve rather than a single first/later split.
    def first_exposure_analysis_c3(results, omit_combo, n_exposure_bins=N_EXPOSURE_BINS):
        first_exp_acc, later_omit_acc, other_acc, omitted_acc = [], [], [], []
        first_omit_trial = []
        exposure_curve = [[] for _ in range(n_exposure_bins)]
        for rnd in range(n_seeds):
            res = results[rnd]
            acc_dict, cue_lab_dict, stim_seq_dict = res[3], res[8], res[11]
            seen_omit = False
            omit_count = 0
            first_trial, later_trial, other_trial, omit_trial = [], [], [], []
            exposure_bins = [[] for _ in range(n_exposure_bins)]
            for blk_idx in range(num_real_blocks):
                actual_blk = blk_idx + num_practice_blocks
                acc = acc_dict[actual_blk]
                cues = np.array([cue_lab_dict[actual_blk][trl] for trl in range(len(acc))])
                stim = stim_seq_dict[actual_blk]
                omit_mask = (cues == omit_combo[0]) & (stim == omit_combo[1])
                for trl in range(len(acc)):
                    global_trial = blk_idx * num_trials + trl
                    if omit_mask[trl]:
                        omit_trial.append(acc[trl])
                        if not seen_omit:
                            first_trial.append(acc[trl])
                            first_omit_trial.append(global_trial)
                            seen_omit = True
                        else:
                            later_trial.append(acc[trl])
                        if omit_count < n_exposure_bins:
                            exposure_bins[omit_count].append(acc[trl])
                        omit_count += 1
                    else:
                        other_trial.append(acc[trl])
            if first_trial: first_exp_acc.append(np.mean(first_trial))
            if later_trial: later_omit_acc.append(np.mean(later_trial))
            if other_trial: other_acc.append(np.mean(other_trial))
            if omit_trial: omitted_acc.append(np.mean(omit_trial))
            for i in range(n_exposure_bins):
                if exposure_bins[i]:
                    exposure_curve[i].append(np.mean(exposure_bins[i]))
        exposure_mean = [np.mean(e) if e else np.nan for e in exposure_curve]
        exposure_ster = [np.std(e) / np.sqrt(len(e)) if e else np.nan for e in exposure_curve]
        return {
            'first_exp_acc': first_exp_acc, 'later_acc': later_omit_acc,
            'other_acc': other_acc, 'omitted_acc': omitted_acc,
            'exposure_mean': exposure_mean, 'exposure_ster': exposure_ster,
            'first_omit_trial': first_omit_trial,
        }

    c3 = first_exposure_analysis_c3(results_control3, omit_combo)

    # ---------------- SIGNIFICANCE TESTS ----------------
    # Paired by seed: each seed contributes one matched pair of values (its own
    # accuracy under both conditions, or its own trained-vs-novel / omitted-vs-
    # other split), consistent with the per-seed SEM used for the bars above —
    # seed*block rows would pseudo-replicate correlated trials from one run.
    # Control 1 asks whether these DIFFER (shuffling should hurt); Controls
    # 2/3 ask the opposite -- whether novel/omitted performs EQUIVALENTLY to
    # trained/other -- which is what the TOST fields in each stats dict are
    # for (see paired_comparison's docstring).
    stats_shuffle = paired_comparison(seed_acc_baseline, seed_acc_c1)
    t_shuffle, p_shuffle = stats_shuffle['t'], stats_shuffle['p']
    stats_novel = paired_comparison(c2['trained_acc'], c2['novel_acc'])
    t_novel, p_novel = stats_novel['t'], stats_novel['p']
    stats_omit = paired_comparison(c3['omitted_acc'], c3['other_acc'])
    t_omit, p_omit = stats_omit['t'], stats_omit['p']

    # ---------------- PLOTS ----------------
    fig, ax = plt.subplots(2, 4, figsize=(24, 10))

    ax[0, 0].errorbar(range(num_trials), mean_sat_baseline, yerr=ster_sat_baseline, label='Baseline', color='blue')
    ax[0, 0].errorbar(range(num_trials), mean_sat_c1, yerr=ster_sat_c1, label='Shuffled cues', color='red')
    ax[0, 0].errorbar(range(num_trials), mean_sat_c2, yerr=ster_sat_c2, label='Novel cue restriction', color='green')
    ax[0, 0].errorbar(range(num_trials), mean_sat_c3, yerr=ster_sat_c3, label='Omitted combo', color='orange')
    ax[0, 0].set_xlabel('Trial'); ax[0, 0].set_ylabel('Eigenvalues > 1')
    ax[0, 0].legend(); ax[0, 0].set_title('Rule representation strength')

    labels = ['Baseline', 'Shuffled', 'Novel-cue\nrestriction', 'Omitted\ncombo']
    means = [np.mean(a) for a in [all_acc_baseline, all_acc_c1, all_acc_c2, all_acc_c3]]
    stds = [np.std(a) / np.sqrt(n_seeds) for a in [all_acc_baseline, all_acc_c1, all_acc_c2, all_acc_c3]]
    ax[0, 1].bar(labels, means, yerr=stds, color=['blue', 'red', 'green', 'orange'], alpha=.7)
    ax[0, 1].set_ylim(0, 1.1); ax[0, 1].set_title('Overall accuracy by condition')

    ax[0, 2].errorbar(block_kinds, mean_shuffle_cost, yerr=ster_shuffle_cost, color='red', marker='o')
    ax[0, 2].axhline(0, color='grey', linestyle='--')
    ax[0, 2].set_xlabel('Switch probability'); ax[0, 2].set_ylabel('Baseline - Shuffled accuracy')
    ax[0, 2].set_title('Cue-shuffle cost by block switch rate')

    ax[0, 3].axis('off')

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
    ax[0, 3].text(0.02, 0.98, stats_text, transform=ax[0, 3].transAxes,
                  va='top', ha='left', fontsize=8.5, family='monospace')

    labels2 = ['Trained\ncues', 'Novel cue,\nfirst exposure', 'Novel cue,\nafter learning']
    means2 = [np.mean(c2['trained_acc']), np.mean(c2['first_exp_acc']), np.mean(c2['later_acc'])]
    stds2 = [np.std(c2['trained_acc']) / np.sqrt(len(c2['trained_acc'])),
             np.std(c2['first_exp_acc']) / np.sqrt(len(c2['first_exp_acc'])),
             np.std(c2['later_acc']) / np.sqrt(len(c2['later_acc']))]
    ax[1, 0].bar(labels2, means2, yerr=stds2, color=['seagreen', 'salmon', 'lightgreen'])
    ax[1, 0].set_ylim(0, 1.1); ax[1, 0].set_title('Control 2: novel-cue transfer over time')

    labels3 = ['Other\ncombos', 'Omitted combo,\nfirst exposure', 'Omitted combo,\nafter learning']
    means3 = [np.mean(c3['other_acc']), np.mean(c3['first_exp_acc']), np.mean(c3['later_acc'])]
    stds3 = [np.std(c3['other_acc']) / np.sqrt(len(c3['other_acc'])),
             np.std(c3['first_exp_acc']) / np.sqrt(len(c3['first_exp_acc'])),
             np.std(c3['later_acc']) / np.sqrt(len(c3['later_acc']))]
    ax[1, 1].bar(labels3, means3, yerr=stds3, color=['navajowhite', 'salmon', 'darkorange'])
    ax[1, 1].set_ylim(0, 1.1); ax[1, 1].set_title('Control 3: omitted-combo transfer over time')

    exposure_x = np.arange(1, N_EXPOSURE_BINS + 1)
    ax[1, 2].errorbar(exposure_x, c2['exposure_mean'], yerr=c2['exposure_ster'], color='green', marker='o')
    ax[1, 2].axhline(np.mean(c2['trained_acc']), color='seagreen', linestyle='--', label='Trained-cue mean')
    ax[1, 2].set_xticks(exposure_x); ax[1, 2].set_ylim(0, 1.1)
    ax[1, 2].set_xlabel('Novel-cue exposure #'); ax[1, 2].set_ylabel('Accuracy')
    ax[1, 2].set_title('Control 2: novel-cue learning curve'); ax[1, 2].legend()

    ax[1, 3].errorbar(exposure_x, c3['exposure_mean'], yerr=c3['exposure_ster'], color='orange', marker='o')
    ax[1, 3].axhline(np.mean(c3['other_acc']), color='darkorange', linestyle='--', label='Other-combo mean')
    ax[1, 3].set_xticks(exposure_x); ax[1, 3].set_ylim(0, 1.1)
    ax[1, 3].set_xlabel('Omitted-combo exposure #'); ax[1, 3].set_ylabel('Accuracy')
    ax[1, 3].set_title('Control 3: omitted-combo learning curve'); ax[1, 3].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'full_control_comparison.png'), dpi=300)
    plt.show()

    # ---------------- SUMMARY ----------------
    def _sem(values):
        values = np.asarray(values, dtype=float)
        return float(np.std(values) / np.sqrt(len(values))) if len(values) else float('nan')

    def _print_stats(label, s):
        eq = 'EQUIVALENT' if s['equivalent'] else 'not established as equivalent'
        print(f'  -> paired t-test {label}: t={s["t"]:.3f}, p={s["p"]:.5f}')
        print(f'  -> mean diff (95% CI): {s["mean_diff"]:+.3f} [{s["ci_low"]:+.3f}, {s["ci_high"]:+.3f}], '
              f"Cohen's d={s['cohens_d']:.3f}")
        print(f"  -> TOST equivalence test (bound=+/-{s['equivalence_bound']:.2f}): "
              f"p={s['tost_p']:.5f} -> {eq}")

    total_test_trials = num_real_blocks * num_trials
    print("\n=== SUMMARY ===")
    print(f"Baseline accuracy:              {np.mean(all_acc_baseline):.3f}")
    print(f"Shuffled-cue accuracy:          {np.mean(all_acc_c1):.3f}")
    print(f"Novel-cue-restriction accuracy: {np.mean(all_acc_c2):.3f}")
    print(f"Omitted-combo accuracy:         {np.mean(all_acc_c3):.3f}")
    _print_stats('baseline vs shuffled', stats_shuffle)
    print()
    print(f"Control 2 - trained cues:                  {np.mean(c2['trained_acc']):.3f}")
    print(f"Control 2 - novel cue, FIRST exposure:     {np.mean(c2['first_exp_acc']):.3f}")
    print(f"Control 2 - novel cue, after learning:      {np.mean(c2['later_acc']):.3f}")
    print(f"Control 2 - novel cue, ALL exposures:      {np.mean(c2['novel_acc']):.3f}")
    _print_stats('trained vs novel', stats_novel)
    print(f"  -> mean trial of first A2 exposure:       {np.mean(c2['first_A2_trial']):.1f} / {total_test_trials} test trials")
    print(f"  -> mean trial of first B2 exposure:       {np.mean(c2['first_B2_trial']):.1f} / {total_test_trials} test trials")
    print(f"  -> accuracy by novel-cue exposure # (1-{N_EXPOSURE_BINS}): {[round(float(v), 3) for v in c2['exposure_mean']]}")
    print()
    print(f"Control 3 - other combos:                  {np.mean(c3['other_acc']):.3f}")
    print(f"Control 3 - omitted combo, FIRST exposure: {np.mean(c3['first_exp_acc']):.3f}")
    print(f"Control 3 - omitted combo, after learning:  {np.mean(c3['later_acc']):.3f}")
    print(f"Control 3 - omitted combo, ALL exposures:  {np.mean(c3['omitted_acc']):.3f}")
    _print_stats('omitted vs other', stats_omit)
    print(f"  -> mean trial of first omitted exposure:  {np.mean(c3['first_omit_trial']):.1f} / {total_test_trials} test trials")
    print(f"  -> accuracy by omitted-combo exposure # (1-{N_EXPOSURE_BINS}): {[round(float(v), 3) for v in c3['exposure_mean']]}")

    print('\nNote: TOST equivalence is only a meaningful claim for Controls 2/3 '
          '("performs just as well as"); Control 1 is testing for a difference, not equivalence.')

    # ---------------- JSON SUMMARY EXPORT (for compare_all_models.py) ----------------
    summary = {
        'model': 'updated_model_v1',
        'n_seeds': n_seeds,
        'equivalence_bound': EQUIVALENCE_BOUND,
        'baseline_accuracy_mean': float(np.mean(seed_acc_baseline)),
        'baseline_accuracy_sem': _sem(seed_acc_baseline),
        'control1_accuracy_mean': float(np.mean(seed_acc_c1)),
        'control1_accuracy_sem': _sem(seed_acc_c1),
        'control1_t': stats_shuffle['t'], 'control1_p': stats_shuffle['p'],
        'control1_ci': [stats_shuffle['ci_low'], stats_shuffle['ci_high']],
        'control1_cohens_d': stats_shuffle['cohens_d'],
        'control1_tost_p': stats_shuffle['tost_p'], 'control1_equivalent': stats_shuffle['equivalent'],
        'control2_trained_mean': float(np.mean(c2['trained_acc'])),
        'control2_trained_sem': _sem(c2['trained_acc']),
        'control2_novel_mean': float(np.mean(c2['novel_acc'])),
        'control2_novel_sem': _sem(c2['novel_acc']),
        'control2_t': stats_novel['t'], 'control2_p': stats_novel['p'],
        'control2_ci': [stats_novel['ci_low'], stats_novel['ci_high']],
        'control2_cohens_d': stats_novel['cohens_d'],
        'control2_tost_p': stats_novel['tost_p'], 'control2_equivalent': stats_novel['equivalent'],
        'control3_other_mean': float(np.mean(c3['other_acc'])),
        'control3_other_sem': _sem(c3['other_acc']),
        'control3_omitted_mean': float(np.mean(c3['omitted_acc'])),
        'control3_omitted_sem': _sem(c3['omitted_acc']),
        'control3_t': stats_omit['t'], 'control3_p': stats_omit['p'],
        'control3_ci': [stats_omit['ci_low'], stats_omit['ci_high']],
        'control3_cohens_d': stats_omit['cohens_d'],
        'control3_tost_p': stats_omit['tost_p'], 'control3_equivalent': stats_omit['equivalent'],
    }
    with open(os.path.join(output_dir, 'controls_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {os.path.join(output_dir, 'controls_summary.json')}")
