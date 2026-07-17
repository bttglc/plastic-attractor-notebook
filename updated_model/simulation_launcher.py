"""
Basic (non-TMS) launcher for plastic attractor model of feature-selective attention

This script runs the basic network simulation and the weight matrix
eigenvalue analysis, extracted from plasticattracto_centralscript.py.

@author: Christopher Whyte
@extension: NMA team

"""""

import os
import numpy as np
import matplotlib.pyplot as plt
# simulate network
from attractor_rnn import plasticattractor_sim

# output dir for figures
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(output_dir, exist_ok=True)

# trials per real block (must be a multiple of 4)
num_trials = 48

# trials per practice block (must be a multiple of 8: 2 cues x 4 stimuli)
practice_trials = 48

# probability of a rule switch on each trial, one entry per real block
switch_probs = [0.125, 0.25, 0.5, 0.75] * 2

# blocks 0 and 1 are practice, the rest are real
num_practice_blocks = 2
num_real_blocks = len(switch_probs)

# number of random seeds
n_seeds = 20

# MODEL VERSION PARAMETERS #
# ===================================================== #

# named parameter sets for model comparison across iterations. each key
# mirrors a keyword argument accepted by plasticattractor_sim; add further
# entries here to compare against the original Whyte model parameters
model_versions = {
    'whyte_original': {
        'alpha_cc_inhib': -.45, 'alpha_cc_decay': 1, 'alpha_cf_gain': .08,
        'alpha_ff_inhib': -.28, 'alpha_ff_decay': .73, 'alpha_fc_gain': .04,
        'beta': .175, 'lr_short': .02, 'lr_long': .0002,
        'w_short_bound': 1, 'w_long_bound': .2,
        'w_short_weight': 1, 'w_long_weight': 1,
    },
}

# select which parameter set this run uses
active_version = 'whyte_original'

# BASIC SIMULATION #
# ===================================================== #

c_lab = {}; s_lab = {}; r_lab = {}; cue_lab = {}; trans_lab = {}; conj = {}; feat = {};
choice = {}; acc = {}; rt = {}; weights = {}

for rnd in range(n_seeds):
    conj[rnd], feat[rnd], choice[rnd], acc[rnd], rt[rnd],\
    c_lab[rnd], s_lab[rnd], r_lab[rnd], cue_lab[rnd], trans_lab[rnd], weights[rnd]\
    = plasticattractor_sim(num_trials, rnd, switch_probs, practice_trials, TMS_sim = False, TMS_start = None,
                            **model_versions[active_version])
    print('Basic task simulation progress', ((rnd+1)/n_seeds)*100, '%')

# WEIGHT MATRIX EIGENVALUE ANALYSIS #
# ===================================================== #

# real blocks only: the '4 eigenvalues > 1' claim is about the post-instruction
# network holding both rules at once, so the practice blocks (where only one
# rule exists) are analysed separately below
# np.linalg.eig(W@W.T) returns one eigenvalue per row/col of W@W.T, i.e. per
# feature unit - not per conjunction unit. rank(W@W.T) is capped at
# num_conjunction, so most of these come out near zero, but the eigenvalue
# array itself is num_features long
num_features_eig = weights[0][0, num_practice_blocks].shape[0]

sat_vals = np.zeros([n_seeds, num_trials, num_real_blocks])
all_eigvals = np.zeros([n_seeds, num_trials, num_real_blocks, num_features_eig])
for rnd in range(n_seeds):
    for blk in range(num_real_blocks):
        for trl in range(num_trials):
            W = weights[rnd][trl, blk + num_practice_blocks]
            [vals, vecs] = np.linalg.eig(W@W.T)
            # number of saturating eigen values (i.e. >1)
            sat_vals[rnd,trl,blk] = np.sum(vals>1)
            # magnitude of every eigenvalue (not just the largest), for the
            # by-block-kind spread plot
            all_eigvals[rnd,trl,blk,:] = np.abs(vals)

mean_sat_vals = np.mean(np.mean(sat_vals,0),1)
ster_sat_vals = np.std(np.mean(sat_vals,0),1)/np.sqrt(n_seeds)

# PRACTICE BLOCK EIGENVALUE ANALYSIS #
# ===================================================== #

# same analysis over the two practice blocks, kept separate so the climb from
# ~2 to ~4 saturating eigenvalues as the second rule is taught stays visible
prac_sat_vals = np.zeros([n_seeds, practice_trials, num_practice_blocks])
for rnd in range(n_seeds):
    for blk in range(num_practice_blocks):
        for trl in range(practice_trials):
            W = weights[rnd][trl, blk]
            [vals, vecs] = np.linalg.eig(W@W.T)
            prac_sat_vals[rnd,trl,blk] = np.sum(vals>1)

# concatenate the practice blocks into one learning curve, block 0 then block 1
mean_prac_sat_vals = np.mean(prac_sat_vals,0).T.reshape(-1)
ster_prac_sat_vals = (np.std(prac_sat_vals,0)/np.sqrt(n_seeds)).T.reshape(-1)

# EIGENVALUES, SWITCH COST, AND INCONGRUENCE COST BY BLOCK KIND #
# ===================================================== #

# block kind = the switch probability a real block was run at. switch_probs
# cycles [0.125, 0.25, 0.5, 0.75] twice, so each kind has 2 blocks behind it
block_kinds = sorted(set(switch_probs))
switch_probs_arr = np.array(switch_probs)

# eigenvalues by block kind #
# ------------------- #

mean_sat_by_kind = np.zeros(len(block_kinds))
ster_sat_by_kind = np.zeros(len(block_kinds))
# pooled (not summarised) eigenvalue magnitudes per kind, for the boxplot -
# pools seeds, trials, both blocks of that kind, and all conjunction units
eigmag_by_kind = []

for k, kind in enumerate(block_kinds):
    kind_blocks = np.where(switch_probs_arr == kind)[0]

    sat_sub = sat_vals[:,:,kind_blocks]
    per_seed_mean = np.mean(sat_sub, axis=(1,2))
    mean_sat_by_kind[k] = np.mean(per_seed_mean)
    ster_sat_by_kind[k] = np.std(per_seed_mean)/np.sqrt(n_seeds)

    eigmag_by_kind.append(all_eigvals[:,:,kind_blocks,:].flatten())

# switch cost and incongruence cost by block kind #
# ------------------- #

# switch cost: rule-switch trials (trans_lab==2) vs full-repeat trials
# (trans_lab==0). cue-switch/rule-repeat trials (trans_lab==1) are excluded
# from both groups so the contrast isolates rule-switching specifically.
# RT costs use correct trials only, matching correct_rt in
# plasticattractor_behaviouralanalysis.py. costs are signed so that a
# positive number always means the harder condition (switch/incongruent)
# was worse.
switch_cost_rt = np.zeros([n_seeds, len(block_kinds)])
switch_cost_acc = np.zeros([n_seeds, len(block_kinds)])
incong_cost_rt = np.zeros([n_seeds, len(block_kinds)])
incong_cost_acc = np.zeros([n_seeds, len(block_kinds)])

# pure (non-differenced) RT and accuracy per trial type, computed alongside
# the costs above since they reuse the same rt_kind/acc_kind/trans_kind/
# congruent_idx groupings
repeat_rt = np.zeros([n_seeds, len(block_kinds)])
switch_rt = np.zeros([n_seeds, len(block_kinds)])
congruent_rt = np.zeros([n_seeds, len(block_kinds)])
incongruent_rt = np.zeros([n_seeds, len(block_kinds)])
repeat_acc = np.zeros([n_seeds, len(block_kinds)])
switch_acc = np.zeros([n_seeds, len(block_kinds)])
congruent_acc = np.zeros([n_seeds, len(block_kinds)])
incongruent_acc = np.zeros([n_seeds, len(block_kinds)])

for rnd in range(n_seeds):
    for k, kind in enumerate(block_kinds):
        kind_blocks = np.where(switch_probs_arr == kind)[0]

        rt_kind = np.concatenate([rt[rnd][blk + num_practice_blocks] for blk in kind_blocks])
        acc_kind = np.concatenate([acc[rnd][blk + num_practice_blocks] for blk in kind_blocks])
        trans_kind = np.concatenate([np.array(list(trans_lab[rnd][blk + num_practice_blocks].values())) for blk in kind_blocks])
        c_kind = np.concatenate([np.array(list(c_lab[rnd][blk + num_practice_blocks].values())) for blk in kind_blocks])
        s_kind = np.concatenate([np.array(list(s_lab[rnd][blk + num_practice_blocks].values())) for blk in kind_blocks])

        correct = acc_kind == 1

        # switch cost
        switch_idx = trans_kind == 2
        repeat_idx = trans_kind == 0
        switch_cost_rt[rnd,k] = np.mean(rt_kind[switch_idx & correct]) - np.mean(rt_kind[repeat_idx & correct])
        switch_cost_acc[rnd,k] = np.mean(acc_kind[repeat_idx]) - np.mean(acc_kind[switch_idx])

        # incongruence cost (congruent = green square, blue circle)
        congruent_idx = np.logical_or(np.logical_and(c_kind==1,s_kind==1), np.logical_and(c_kind==2,s_kind==2))
        incongruent_idx = ~congruent_idx
        incong_cost_rt[rnd,k] = np.mean(rt_kind[incongruent_idx & correct]) - np.mean(rt_kind[congruent_idx & correct])
        incong_cost_acc[rnd,k] = np.mean(acc_kind[congruent_idx]) - np.mean(acc_kind[incongruent_idx])

        # pure RT (correct trials only) and accuracy per trial type
        repeat_rt[rnd,k] = np.mean(rt_kind[repeat_idx & correct])
        switch_rt[rnd,k] = np.mean(rt_kind[switch_idx & correct])
        congruent_rt[rnd,k] = np.mean(rt_kind[congruent_idx & correct])
        incongruent_rt[rnd,k] = np.mean(rt_kind[incongruent_idx & correct])
        repeat_acc[rnd,k] = np.mean(acc_kind[repeat_idx])
        switch_acc[rnd,k] = np.mean(acc_kind[switch_idx])
        congruent_acc[rnd,k] = np.mean(acc_kind[congruent_idx])
        incongruent_acc[rnd,k] = np.mean(acc_kind[incongruent_idx])

mean_switch_cost_rt = np.mean(switch_cost_rt,0)
ster_switch_cost_rt = np.std(switch_cost_rt,0)/np.sqrt(n_seeds)
mean_switch_cost_acc = np.mean(switch_cost_acc,0)
ster_switch_cost_acc = np.std(switch_cost_acc,0)/np.sqrt(n_seeds)

mean_incong_cost_rt = np.mean(incong_cost_rt,0)
ster_incong_cost_rt = np.std(incong_cost_rt,0)/np.sqrt(n_seeds)
mean_incong_cost_acc = np.mean(incong_cost_acc,0)
ster_incong_cost_acc = np.std(incong_cost_acc,0)/np.sqrt(n_seeds)

mean_repeat_rt = np.mean(repeat_rt,0)
ster_repeat_rt = np.std(repeat_rt,0)/np.sqrt(n_seeds)
mean_switch_rt = np.mean(switch_rt,0)
ster_switch_rt = np.std(switch_rt,0)/np.sqrt(n_seeds)
mean_congruent_rt = np.mean(congruent_rt,0)
ster_congruent_rt = np.std(congruent_rt,0)/np.sqrt(n_seeds)
mean_incongruent_rt = np.mean(incongruent_rt,0)
ster_incongruent_rt = np.std(incongruent_rt,0)/np.sqrt(n_seeds)

mean_repeat_acc = np.mean(repeat_acc,0)
ster_repeat_acc = np.std(repeat_acc,0)/np.sqrt(n_seeds)
mean_switch_acc = np.mean(switch_acc,0)
ster_switch_acc = np.std(switch_acc,0)/np.sqrt(n_seeds)
mean_congruent_acc = np.mean(congruent_acc,0)
ster_congruent_acc = np.std(congruent_acc,0)/np.sqrt(n_seeds)
mean_incongruent_acc = np.mean(incongruent_acc,0)
ster_incongruent_acc = np.std(incongruent_acc,0)/np.sqrt(n_seeds)

# PERFORMANCE EVOLUTION ACROSS BLOCKS #
# ===================================================== #

# overall (not split by trial type) RT and accuracy per block index, across
# practice and real blocks together, to show how performance evolves over
# the course of the whole session. RT uses correct trials only, matching
# correct_rt in plasticattractor_behaviouralanalysis.py
num_blocks_total = num_practice_blocks + num_real_blocks

block_rt = np.zeros([n_seeds, num_blocks_total])
block_acc = np.zeros([n_seeds, num_blocks_total])

for rnd in range(n_seeds):
    for blk in range(num_blocks_total):
        rt_blk = rt[rnd][blk]
        acc_blk = acc[rnd][blk]
        correct = acc_blk == 1
        block_rt[rnd,blk] = np.mean(rt_blk[correct])
        block_acc[rnd,blk] = np.mean(acc_blk)

mean_block_rt = np.mean(block_rt,0)
ster_block_rt = np.std(block_rt,0)/np.sqrt(n_seeds)
mean_block_acc = np.mean(block_acc,0)
ster_block_acc = np.std(block_acc,0)/np.sqrt(n_seeds)

# EIGEN VALUE PLOTS #
# ===================================================== #

plt.rcParams['font.size'] = 12

trial_index = np.arange(0,num_trials,1)

# number of saturating eigenvalues vs trial
fig = plt.figure()
plt.errorbar(trial_index,mean_sat_vals, yerr=ster_sat_vals, color = 'black', capsize=3, linestyle="-",marker="o")
plt.xlabel("Trial")
plt.ylabel("Eigenvalues > 1")
plt.xticks(trial_index[::4])
fig.savefig(os.path.join(output_dir, 'eigenvalues_above_1_vs_trial.png'), format='png', dpi=1200)

# practice learning curve, with the boundary between the two practice blocks
prac_index = np.arange(0,practice_trials*num_practice_blocks,1)
fig = plt.figure()
plt.errorbar(prac_index,mean_prac_sat_vals, yerr=ster_prac_sat_vals, color = 'black', capsize=3, linestyle="-",marker="o")
plt.axvline(practice_trials-.5, color = 'grey', linestyle="--")
plt.xlabel("Practice trial")
plt.ylabel("Eigenvalues > 1")
plt.xticks(prac_index[::8])
fig.savefig(os.path.join(output_dir, 'practice_eigenvalues_above_1.png'), format='png', dpi=1200)

# BLOCK KIND PLOTS #
# ===================================================== #

kind_labels = [str(kind) for kind in block_kinds]

# all eigenvalue magnitudes per block kind
fig, ax = plt.subplots()
ax.boxplot(eigmag_by_kind, tick_labels=kind_labels, showfliers=False)
ax.set_xlabel("Switch probability (block kind)")
ax.set_ylabel("Eigenvalue magnitude")
fig.savefig(os.path.join(output_dir, 'eigenvalue_magnitudes_by_block_kind.png'), format='png', dpi=1200)

# count of amplifying eigenvalues (>1) per block kind
fig = plt.figure()
plt.errorbar(block_kinds,mean_sat_by_kind, yerr=ster_sat_by_kind, color = 'black', capsize=3, linestyle="-",marker="o")
plt.xlabel("Switch probability (block kind)")
plt.ylabel("Amplifying eigenvalues (>1)")
fig.savefig(os.path.join(output_dir, 'amplifying_eigenvalues_by_block_kind.png'), format='png', dpi=1200)

# switch cost (RT and accuracy) per block kind
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].errorbar(block_kinds,mean_switch_cost_rt, yerr=ster_switch_cost_rt, color = 'black', capsize=3, linestyle="-",marker="o")
ax[0].set_xlabel("Switch probability (block kind)")
ax[0].set_ylabel("Switch cost: RT (time-steps)")
ax[1].errorbar(block_kinds,mean_switch_cost_acc, yerr=ster_switch_cost_acc, color = 'black', capsize=3, linestyle="-",marker="o")
ax[1].set_xlabel("Switch probability (block kind)")
ax[1].set_ylabel("Switch cost: accuracy")
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'switch_cost_by_block_kind.png'), format='png', dpi=1200)

# incongruence cost (RT and accuracy) per block kind
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].errorbar(block_kinds,mean_incong_cost_rt, yerr=ster_incong_cost_rt, color = 'black', capsize=3, linestyle="-",marker="o")
ax[0].set_xlabel("Switch probability (block kind)")
ax[0].set_ylabel("Incongruence cost: RT (time-steps)")
ax[1].errorbar(block_kinds,mean_incong_cost_acc, yerr=ster_incong_cost_acc, color = 'black', capsize=3, linestyle="-",marker="o")
ax[1].set_xlabel("Switch probability (block kind)")
ax[1].set_ylabel("Incongruence cost: accuracy")
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'incongruence_cost_by_block_kind.png'), format='png', dpi=1200)

# repeat vs switch RT and accuracy per block kind
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].errorbar(block_kinds,mean_repeat_rt, yerr=ster_repeat_rt, color = 'black', capsize=3, linestyle="-",marker="o", label='Repeat')
ax[0].errorbar(block_kinds,mean_switch_rt, yerr=ster_switch_rt, color = 'black', capsize=3, linestyle="--",marker="s", label='Switch')
ax[0].set_xlabel("Switch probability (block kind)")
ax[0].set_ylabel("RT (time-steps)")
ax[0].legend()
ax[1].errorbar(block_kinds,mean_repeat_acc, yerr=ster_repeat_acc, color = 'black', capsize=3, linestyle="-",marker="o", label='Repeat')
ax[1].errorbar(block_kinds,mean_switch_acc, yerr=ster_switch_acc, color = 'black', capsize=3, linestyle="--",marker="s", label='Switch')
ax[1].set_xlabel("Switch probability (block kind)")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'repeat_switch_rt_acc_by_block_kind.png'), format='png', dpi=1200)

# congruent vs incongruent RT and accuracy per block kind
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].errorbar(block_kinds,mean_congruent_rt, yerr=ster_congruent_rt, color = 'black', capsize=3, linestyle="-",marker="o", label='Congruent')
ax[0].errorbar(block_kinds,mean_incongruent_rt, yerr=ster_incongruent_rt, color = 'black', capsize=3, linestyle="--",marker="s", label='Incongruent')
ax[0].set_xlabel("Switch probability (block kind)")
ax[0].set_ylabel("RT (time-steps)")
ax[0].legend()
ax[1].errorbar(block_kinds,mean_congruent_acc, yerr=ster_congruent_acc, color = 'black', capsize=3, linestyle="-",marker="o", label='Congruent')
ax[1].errorbar(block_kinds,mean_incongruent_acc, yerr=ster_incongruent_acc, color = 'black', capsize=3, linestyle="--",marker="s", label='Incongruent')
ax[1].set_xlabel("Switch probability (block kind)")
ax[1].set_ylabel("Accuracy")
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'congruent_incongruent_rt_acc_by_block_kind.png'), format='png', dpi=1200)

# PERFORMANCE EVOLUTION PLOT #
# ===================================================== #

# overall RT and accuracy per block index, with the practice/real boundary
# marked, same convention as the practice eigenvalue plot above
block_index = np.arange(0,num_blocks_total,1)
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].errorbar(block_index,mean_block_rt, yerr=ster_block_rt, color = 'black', capsize=3, linestyle="-",marker="o")
ax[0].axvline(num_practice_blocks-.5, color = 'grey', linestyle="--")
ax[0].set_xlabel("Block")
ax[0].set_ylabel("RT (time-steps)")
ax[0].set_xticks(block_index)
ax[1].errorbar(block_index,mean_block_acc, yerr=ster_block_acc, color = 'black', capsize=3, linestyle="-",marker="o")
ax[1].axvline(num_practice_blocks-.5, color = 'grey', linestyle="--")
ax[1].set_xlabel("Block")
ax[1].set_ylabel("Accuracy")
ax[1].set_xticks(block_index)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'performance_evolution_by_block.png'), format='png', dpi=1200)

plt.show()
