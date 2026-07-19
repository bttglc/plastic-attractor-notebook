"""
Plastic attractor network simulation script

Cued task-switching extension: the network is instructed once, up front, in two
supervised practice blocks (one rule each). After that the rule varies trial by
trial and is signalled only by a cue presented alongside the stimulus.

@author: Christopher Whyte
@extension: NMA team

"""""

def plasticattractor_sim(num_trials, rnd_seed, switch_probs, practice_trials, TMS_sim, TMS_start,
                          alpha_cc_inhib=-.45, alpha_cc_decay=1, alpha_cf_gain=.08,
                          alpha_ff_inhib=-.28, alpha_ff_decay=.73, alpha_fc_gain=.04,
                          beta=.175, lr_short=.02, lr_long=.0002,
                          w_short_bound=1, w_long_bound=.2,
                          w_short_weight=1, w_long_weight=1):

    import numpy as np

    # seed random number generator
    np.random.seed(rnd_seed)

    # %% params

    # MODEL PARAMETERS #
    # ===================================================== #

    # number of feature units, ordered
    # [green, blue, square, circle, cueA1, cueA2, cueB1, cueB2, action1, action2]
    #    0     1      2       3       4      5      6      7       8        9
    # cueA1/cueA2 both mean 'attend colour', cueB1/cueB2 both mean 'attend
    # shape', so cue identity and abstract rule are dissociable
    num_features = 10

    # number of conjection units
    # this caps rank(W@W.T), so it is also the ceiling on the number of
    # saturating eigenvalues: 4 conjunction units can support at most 4
    # rule x response attractors, which is exactly what the task demands
    num_conjunction = 4

    alpha = np.array([alpha_cc_inhib, alpha_cc_decay, alpha_cf_gain, alpha_ff_inhib, alpha_ff_decay, alpha_fc_gain])

    # baseline activity/lateral inhibition threshold

    # learning rate for hebbian updates
    gamma = lr_short

    # the two action units within the feature layer, named once so the readout
    # below never has to repeat the index
    action_idx = slice(8, num_features)

    # FEATURE UNITS LOCAL EXCITATION / INHIBITION #
    # ===================================================== #

    # feature -> feature weights. This is a num_features x num_features matrix
    # with blocks down the diagonal so that only neurons within each feature
    # group compete via lateral inhibition.

    # step 1: alpha[4] * eye puts self excitation/decay (.73) on the diagonal only

    # step 2: alpha[3] * inhibition_mask puts mutual inhibition (-.28) within
    # each group and zero between groups, e.g. for the colour group (rows/cols
    # 0-1) this is the 2x2 block
    #   [[-.28, -.28]
    #    [-.28, -.28]]

    # W_ff = step 1 + step 2, so each 2x2 group is [[.45,-.28],[-.28,.45]] and
    # the diagonal is alpha[4]+alpha[3] = .45 throughout

    # note: inhibition is local only, block-diagonal, so a feature unit only
    # competes within its own group, never with units from other groups -
    # contrast with W_cc below, where inhibition is global across all units

    # the 4 cue units form a single group rather than two pairs, so competition
    # is winner-take-all across all cues - only one cue is ever presented per
    # trial, and the two cues of a rule should not be treated as a pair
    feature_groups = [[0, 1], [2, 3], [4, 5, 6, 7], [8, 9]]
    inhibition_mask = np.zeros([num_features, num_features])
    for group in feature_groups:
        inhibition_mask[np.ix_(group, group)] = 1

    W_ff = alpha[4] * np.eye(num_features) + alpha[3] * inhibition_mask

    # CONJUNCTION UNITS LOCAL EXCITATION / INHIBITION #
    # ===================================================== #

    # conjunction -> conjuction weights. This is a num_conjection x
    # num_conjection matrix with alpha(0) + alpha(1) on the diagonal and
    # alpha(0) in the off diagonal elements. When multiplied with a vectors
    # whose elements are [conjuction_neuron - beta] this creates lateral
    # inhibition, or self excitation, depending on whether the conjunction
    # neurons are > or < beta.

    # step 1: alpha[1] * eye(4) puts self excitation (1) on the diagonal only
    #   [[1, 0, 0, 0]
    #    [0, 1, 0, 0]
    #    [0, 0, 1, 0]
    #    [0, 0, 0, 1]]

    # step 2: alpha[0] * ones(4) is a 1D vector (-.45, -.45, -.45, -.45). Adding it
    # to the matrix above broadcasts it onto every row, so -.45 lands on every
    # element, diagonal included
    #   [-.45, -.45, -.45, -.45]

    # W_cc = step 1 + step 2, alpha(0)+alpha(1) on the diagonal, alpha(0) off diagonal
    #   [[.55, -.45, -.45, -.45]
    #    [-.45,  .55, -.45, -.45]
    #    [-.45, -.45,  .55, -.45]
    #    [-.45, -.45, -.45,  .55]]

    # note: inhibition is global here, every conjunction unit inhibits every
    # other conjunction unit equally, so this is winner-take-all across the
    # whole population, not just within pairs like W_ff above

    W_cc = alpha[1]*np.eye(num_conjunction) + alpha[0]*np.ones(num_conjunction)

    # TASK STRUCTURE AND STIMULUS GENERATION #
    # ===================================================== #

    # Cues #
    # ------------------- #

    # the rule cue shown on every trial, analogous to a symbol on screen that
    # tells a participant 'respond by colour' or 'respond by shape' for this
    # trial. it is an EXTERNAL INPUT VALUE, not a connection weight - it gets
    # added directly onto feature_units each timestep, the same role stimuli
    # and isi play. the only actual weights in this model are W_ff, W_cc, and
    # W/w_short/w_long

    # each cue simply turns on its own cue unit and nothing else. crucially the
    # cue says nothing about which button is correct - it only identifies the
    # rule. the network has to recover the response from the cue x stimulus
    # conjunction it learned during practice (via the learned W weights)
    cues = np.zeros([4, num_features])
    cues[0, 4] = 1   # cueA1 -> attend colour
    cues[1, 5] = 1   # cueA2 -> attend colour
    cues[2, 6] = 1   # cueB1 -> attend shape
    cues[3, 7] = 1   # cueB2 -> attend shape

    # maps cue index -> rule, so two different cues signal the same rule
    # rule 0 = attend colour, rule 1 = attend shape
    cue_rule = np.array([0, 0, 1, 1])

    # Stimuli #
    # ------------------- #

    # the 4 possible pictures shown mid-trial: every combination of 2 colours x
    # 2 shapes. each row turns on the matching colour unit and the matching
    # shape unit, e.g. [1,0,1,0,...] = green=1, square=1 = 'green square'.
    # the cue and action units are always left at 0 here - the stimulus itself
    # carries neither the rule nor the answer
    stimuli = np.zeros([4, num_features])
    stimuli[0, :4] = [1, 0, 1, 0]   # green square
    stimuli[1, :4] = [1, 0, 0, 1]   # green circle
    stimuli[2, :4] = [0, 1, 1, 0]   # blue square
    stimuli[3, :4] = [0, 1, 0, 1]   # blue circle

    # Response teaching #
    # ------------------- #

    # drive applied to the 2 action units on practice trials only, pushing the
    # correct/incorrect action unit towards/away from threshold. this is the
    # only thing that distinguishes a practice trial from a real one - the
    # stimulus supplies the features and the cue supplies the rule, so only the
    # response needs teaching
    action_teach = np.array([[1, -1],    # -> button 1 / action 0
                             [-1, 1]])   # -> button 2 / action 1

    # Block structure #
    # ------------------- #

    # blocks 0 and 1 are practice, one rule each, taught with the cue present.
    # the remaining blocks are the real experiment: the rule varies trial by
    # trial and is signalled by the cue alone, with switch_probs giving the
    # per-block probability of a rule switch
    num_practice_blocks = 2
    num_blocks = num_practice_blocks + len(switch_probs)

    # stim period
    stim_period = 400

    # isi
    # negative drive to every feature unit, actively pushes activity down
    # rather than just withdrawing input between events
    isi = np.full(num_features, -1)

    # resp period
    # zero input - the network is left to evolve under its own recurrent
    # dynamics to settle on a response, nothing is imposed externally
    resp_period = np.zeros(num_features)

    def practice_sequence(rule_idx):
        # every combination of this rule's 2 cues x 4 stimuli, tiled to fill the
        # block and shuffled, so both cues are taught on every stimulus.
        # practice_trials must be a multiple of 8
        cue_pool = np.where(cue_rule == rule_idx)[0]
        combos = np.array([[cue, stim] for cue in cue_pool for stim in range(4)])
        combos = np.tile(combos, (practice_trials//8, 1))
        np.random.shuffle(combos)
        rule_seq = np.full(practice_trials, rule_idx)
        return rule_seq, combos[:, 0], combos[:, 1]

    def real_sequence(switch_prob):
        # rule sequence for one real block: first trial random, then switch with
        # probability switch_prob
        rule_seq = np.zeros(num_trials, dtype=int)
        cue_seq = np.zeros(num_trials, dtype=int)
        rule_seq[0] = np.random.randint(2)
        for trl in range(1, num_trials):
            if np.random.rand() < switch_prob:
                rule_seq[trl] = 1 - rule_seq[trl-1]
            else:
                rule_seq[trl] = rule_seq[trl-1]

        # the cue is drawn uniformly from the 2 cues of the active rule, so a
        # rule repeat can still be a cue switch
        for trl in range(num_trials):
            cue_seq[trl] = np.random.choice(np.where(cue_rule == rule_seq[trl])[0])

        # stimulus identity is independent of the rule
        stim_seq = np.tile([0, 1, 2, 3], num_trials//4)
        np.random.shuffle(stim_seq)
        return rule_seq, cue_seq, stim_seq

    def stim_generator(rule_seq, cue_seq, stim_seq, practice):
        # builds the per-timestep input sequence and trial labels for one block.
        # every trial has the same shape: isi, then the stimulus and its rule
        # cue together, then a response window with no input (the network
        # settles on a choice under its own recurrent dynamics), then isi
        stimulus = {}; c_labels = {}; s_labels = {}; r_labels = {}
        cue_labels = {}; trans_labels = {}; ground_truth = {}

        n_trials = len(rule_seq)

        for trl in range(n_trials):
            rule_idx = rule_seq[trl]
            cue_idx = cue_seq[trl]
            cond_idx = stim_seq[trl]

            # ground truth action
            # rule 0 = colour rule: cond 0/1 (green square/circle) -> action 0,
            # cond 2/3 (blue) -> action 1. rule 1 = shape rule: cond 0/2
            # (square) -> action 0, cond 1/3 (circle) -> action 1. this is the
            # answer the network is expected to have learned during practice
            if rule_idx == 0:
                if cond_idx == 0 or cond_idx == 1:
                    correct = 0
                else:
                    correct = 1
            elif rule_idx == 1:
                if cond_idx == 0 or cond_idx == 2:
                    correct = 0
                else:
                    correct = 1

            stim_input = np.zeros([num_features, stim_period])
            for t in range(stim_period):

                # isi
                if t <= 50:
                    stim_input[:, t] = isi
                # present stimulus together with its rule cue
                elif t >= 50 and t <= 100:
                    stim_input[:, t] = stimuli[cond_idx, :] + cues[cue_idx, :]
                # resp period
                elif t >= 100 and t <= 350:
                    stim_input[:, t] = resp_period
                # isi
                elif t >= 350 and t <= 400:
                    stim_input[:, t] = isi

                # on practice trials the correct response is taught alongside
                # the cue. the drive is held across the settling period, not
                # just the stimulus window, because the hebbian update runs
                # every timestep - left unsupervised while it settles the
                # network would reinforce whatever attractor it happened to
                # land on. shorten or lengthen this window if practice over-
                # or under-learns
                if practice and t >= 50 and t <= 350:
                    stim_input[action_idx, t] = action_teach[correct, :]
                    # 0.5 drive to the feature units for the dimension this
                    # rule doesn't use, e.g. shape units while teaching the
                    # colour rule - initially omitted by accident
                    stim_input[feature_groups[1 - rule_idx], t] = 0.5

            stimulus[trl] = stim_input

            # ground truth action
            ground_truth[trl] = correct

            # rules and cues
            r_labels[trl] = rule_idx
            cue_labels[trl] = cue_idx

            # transition type relative to the previous trial
            # 0 = cue repeat, 1 = cue switch but rule repeat, 2 = rule switch,
            # -1 on the first trial of a block, which has no predecessor
            if trl == 0:
                trans_labels[trl] = -1
            elif rule_seq[trl] != rule_seq[trl-1]:
                trans_labels[trl] = 2
            elif cue_seq[trl] != cue_seq[trl-1]:
                trans_labels[trl] = 1
            else:
                trans_labels[trl] = 0

            # labels for classifier
            # colour/shape labels, independent of which rule is active - used
            # later to decode what the network represents rather than what it did.
            # c_labels: 1 = green, 2 = blue. s_labels: 1 = square, 2 = circle
            # (cond_idx 0-3 = green square, green circle, blue square, blue circle,
            # matching the stimuli rows above)
            if cond_idx == 0:
                c_labels[trl] = 1
                s_labels[trl] = 1
            elif cond_idx == 1:
                c_labels[trl] = 1
                s_labels[trl] = 2
            elif cond_idx == 2:
                c_labels[trl] = 2
                s_labels[trl] = 1
            elif cond_idx == 3:
                c_labels[trl] = 2
                s_labels[trl] = 2

        # hands the main simulation loop everything it needs for this block:
        # the per-timestep input sequences (stimulus), the correct answers
        # (ground_truth), and trial labels for later decoding of the
        # network's internal representations (c_labels/s_labels = colour/
        # shape identity, r_labels = which rule was active, cue_labels = which
        # cue signalled it, trans_labels = switch vs repeat)
        return stimulus, ground_truth, c_labels, s_labels, r_labels, cue_labels, trans_labels

    # NETWORK SIMULATION #
    # ===================================================== #

    # initialise dicts to store network activity and labels
    conj_dict = {}; feat_dict = {}; choice_dict = {}; acc_dict = {}; weight_dict = {}
    rt_dict = {}; s_label_dict = {}; r_label_dict = {}; c_label_dict = {}
    cue_label_dict = {}; trans_label_dict = {}

    # initialise conjunction <-> feature weights as two components on
    # different learning timescales: w_short updates fast and bounds high
    # (short-lived, large swings), w_long updates ~100x slower and bounds low
    # (a slow-drifting baseline). W = w_short + w_long is what actually drives
    # the network dynamics. shape is (num_features, num_conjunction), so W is
    # used as-is for the top-down conjunction -> feature drive, and as W.T for
    # the bottom-up feature -> conjunction drive, ie. the same synapses are
    # read in whichever direction the update needs
    w_short = np.random.rand(num_features, num_conjunction)
    w_long = np.random.rand(num_features, num_conjunction)
    W = np.random.rand(num_features, num_conjunction)

    for blk in range(num_blocks):

            # build the trial sequence for this block. practice blocks 0 and 1
            # teach rule 0 and rule 1 respectively; real blocks draw a rule
            # sequence at this block's switch probability
            practice = blk < num_practice_blocks
            if practice:
                rule_seq, cue_seq, stim_seq = practice_sequence(blk)
            else:
                rule_seq, cue_seq, stim_seq = real_sequence(switch_probs[blk - num_practice_blocks])

            n_trials = len(rule_seq)

            # initialise conjunction unit firing rates
            conjunction_units = np.zeros((num_conjunction, stim_period, n_trials))

            # initialise feature unit firing rates
            feature_units = np.zeros((num_features, stim_period, n_trials))

            # initialise choice and reaction time arrays
            choice = np.zeros(n_trials);  rt = np.zeros(n_trials); accuracy = np.zeros(n_trials)

            # call stimulus function
            # stimulus_dict: keys are trl (0 .. n_trials-1), values are the
            # per-timestep input array for that trial, shape
            # (num_features, stim_period), built from isi + stimuli[] + cues[]
            # (stimulus and cue together) + resp_period (zero input, network
            # settles on a choice) + isi, plus the action teaching drive on
            # practice trials
            stimulus_dict, ground_truth, c_labels, s_labels, rule, cue_labels, trans_labels \
                = stim_generator(rule_seq, cue_seq, stim_seq, practice)

            for trl in range(n_trials):

                # grab stimulus for this trial
                stimulus = stimulus_dict[trl]

                # trial-to-trial carryover (not currently done): the t=0 step
                # below wraps to feature_units[:,-1,trl] / conjunction_units[:,-1,trl],
                # which is still zero at this point since this trial's last
                # timestep hasn't been written yet. to carry the previous
                # trial's end state into this trial's t=0 instead of using
                # zero, seed that same slot here before the t loop starts:
                #   if trl > 0:
                #       feature_units[:,-1,trl] = feature_units[:,-1,trl-1]
                #       conjunction_units[:,-1,trl] = conjunction_units[:,-1,trl-1]
                # the update equations below don't need to change, since
                # they already read via the t-1 wraparound

                for t in range(stim_period):

                    # feature neuron
                    # recurrent term (W_ff) + top-down drive from conjunction
                    # units through the plastic weights W, scaled by synaptic
                    # gain alpha[2] + this timestep's external input. note t-1
                    # at t=0 wraps to index -1 (the trial's last timestep),
                    # which is still zero here since the array was just
                    # initialised
                    feature_units[:,t,trl] = beta + W_ff @ (feature_units[:,t-1,trl] - beta) + alpha[2] * \
                                             W @ (conjunction_units[:,t-1,trl] - beta) + stimulus[:,t]

                    # apply non-linearity to feature neuron
                    # clip firing rate to [0,1], guards against runaway
                    # excitation from the recurrent term
                    feature_units[:,t,trl] = np.maximum(0,np.minimum(1,feature_units[:,t,trl]))

                    # conjunction neuron
                    # TMS simulation: clamp all conjunction units to full
                    # activation for the cued window (skipped entirely when
                    # TMS_start == 100, ie. the no-TMS condition), modelling
                    # disruption of that population rather than normal
                    # recurrent processing
                    if TMS_sim == True and t >= TMS_start and t <= 100 and TMS_start!= 100:
                        conjunction_units[:,t,trl] = np.ones(num_conjunction)
                    else:
                        # otherwise same form as the feature update: recurrent
                        # term (W_cc) + bottom-up activity from feature units
                        # through W transposed, scaled by alpha[5], plus small
                        # gaussian noise to break ties between competing units.
                        # W transposed because W is (num_features, num_conjunction);
                        # here the matmul needs to map feature -> conjunction, the
                        # reverse of the feature update below which uses W as-is
                        conjunction_units[:,t,trl] = beta + W_cc @ (conjunction_units[:,t-1,trl] - beta) + alpha[5] \
                                                     * W.T @ (feature_units[:,t-1,trl] - beta) \
                                                     + 0.005 * np.random.randn(num_conjunction,1).T

                    # apply non-linearity to conjection neuron
                    conjunction_units[:,t,trl] = np.maximum(0,np.minimum(1,conjunction_units[:,t,trl]))

                    # calculate delta term for weight update
                    # covariance-style hebbian term: units on the same side of
                    # baseline (both above or both below) strengthen their
                    # connection, units on opposite sides weaken it
                    delta_w = np.outer((feature_units[:,t,trl] - beta),(conjunction_units[:,t,trl] - beta))

                    # weight update
                    w_short = np.maximum(0,np.minimum(w_short_bound, w_short + gamma * delta_w))
                    w_long  = np.maximum(0,np.minimum(w_long_bound, w_long  + lr_long * delta_w)) # evolves at 2 orders of magnitude the rate
                    W = w_short_weight*w_short + w_long_weight*w_long

                # find max activated "action" feature after stimulus offset (offset is padded by +10 timesteps)
                # action_idx selects just the 2 action units (indices 8 and 9)
                max_action = np.amax(feature_units[action_idx,110:,trl])
                # threshold rather than an exact max match, so the first sample
                # near the peak counts even if it never hits the max exactly
                threshold = 98*(max_action/100)
                # rows = which action unit (0 or 1), columns = timestep since t=110
                action_index = np.argwhere(feature_units[action_idx,110:,trl] > threshold)

                # choice
                # first threshold crossing = earliest (action unit, timestep) pair
                choice[trl] = action_index[0,0]

                # reaction time
                rt[trl] = action_index[0,1]

                # accuracy
                if choice[trl] == ground_truth[trl]:
                    accuracy[trl] = 1
                else:
                    accuracy[trl] = 0

                # snapshot of the current weight matrix, for tracking learning across trials
                weight_dict[trl,blk] = W

            # store trial information in relevent dictionaries
            conj_dict[blk] = conjunction_units
            feat_dict[blk] = feature_units
            choice_dict[blk] = choice
            acc_dict[blk] = accuracy
            rt_dict[blk] = rt
            c_label_dict[blk] = c_labels
            s_label_dict[blk] = s_labels
            r_label_dict[blk] = rule
            cue_label_dict[blk] = cue_labels
            trans_label_dict[blk] = trans_labels
            weight_dict[blk] = W

    return conj_dict, feat_dict, choice_dict, acc_dict, rt_dict, c_label_dict, s_label_dict, \
           r_label_dict, cue_label_dict, trans_label_dict, weight_dict
