"""The plastic-attractor network and its one-step update equations.

This module knows how activity and Hebbian weights change. It deliberately knows
nothing about blocks, stimuli, cues, reaction times, or accuracy.

It is a near-verbatim copy of published_model/plastic_attractor/model.py. The
dynamics are vocabulary-agnostic, so the only differences are that the feature
count now follows the ten-unit Feature enum in task.py, and that the fast/slow
weights can be blended with per-timescale weights (defaults reproduce the
published behaviour exactly).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .task import build_vocabulary


# These dataclasses are named containers. They hold settings or recorded values.
@dataclass(frozen=True)
class ModelParameters:
    """Named parameters for the two coupled neural populations.

    The defaults use the values from the authors' public repository. Parameters
    can be overridden directly when we later test new hypotheses.
    """

    number_of_conjunction_units: int = 4
    num_cues_per_rule: int = 2

    baseline_activity: float = 0.175

    conjunction_lateral_weight: float = -0.45
    conjunction_self_weight: float = 1.00
    feature_lateral_weight: float = -0.28
    feature_self_weight: float = 0.73

    conjunction_to_feature_gain: float = 0.08
    feature_to_conjunction_gain: float = 0.04

    # gating units: an optional third population of inhibitory interneurons that
    # learn, from the cue, to suppress the task-irrelevant colour/shape
    # dimension -- the same mechanism on instruction and real trials alike,
    # since both present the stimulus identically (see task.py's
    # gate_target_indices for the hardcoded per-gate suppression targets). 0
    # disables them entirely, leaving the published dynamics bit-for-bit intact.
    number_of_gating_units: int = 0
    cue_to_gating_gain: float = 1.0
    # entered with a minus sign -> inhibition. Well above
    # conjunction_to_feature_gain (0.08): fighting a feature's own recurrent
    # settling dynamics (feature_self_weight=0.73) over the long coasting
    # period after the stimulus window ends needs more pull than the small
    # top-down conjunction term does.
    gating_to_feature_gain: float = 0.4
    # fixed (not learned), multiplicative-gain pathway from a gate onto its
    # OWN dimension's rows (colour gate -> colour rows, shape gate -> shape
    # rows; task.py's gate_relevant_target_indices), on top of the plastic
    # suppression above -- see step()'s gating_enhancement for why it scales
    # the feature's own current activity rather than adding a flat constant.
    # 0.0 (default) disables it, leaving the feature update bit-identical to
    # before it existed. Exploratory: tests whether directly boosting the
    # relevant dimension's sensitivity helps the network commit to the
    # correct conjunction unit, since suppression alone is arithmetically
    # capped against the stimulus's external floor (see gated-attractor
    # model_outline.md's open questions) and the relevant feature's own
    # settled activity is often well below ceiling.
    gating_to_relevant_feature_gain: float = 0.0
    # gate weights blend a fast and a slow timescale, exactly like fast/slow
    # feature-conjunction weights above: fast tracks the teaching drive quickly
    # but is also what reward-gated real-block trials perturb trial by trial;
    # slow moves ~100x slower, so it low-pass-filters that trial-to-trial
    # fluctuation and anchors the rule -> gate mapping learned during
    # instruction instead of losing it.
    gating_fast_learning_rate: float = 0.02
    gating_slow_learning_rate: float = 0.0002
    gating_maximum_fast_weight: float = 1.0
    gating_maximum_slow_weight: float = 0.2
    gating_fast_weight_blend: float = 1.0
    gating_slow_weight_blend: float = 1.0
    # the eligibility trace decays each step (Fremaux & Gerstner 2016) rather
    # than summing flat over the whole ~400-step trial, so the ~50-step cue
    # presentation dominates it instead of being swamped by the long isi and
    # response-settling tail (never accumulated at all, in fact -- see
    # step()'s cue_signal_active, gated to stimulus_window). ~0.98 gives an
    # effective memory of ~50 steps.
    gating_trace_decay: float = 0.98
    # winner-take-all competition between gates (same idiom as the conjunction
    # units), but self_weight is raised well past 1.0 (unlike the conjunction
    # units' 1.00): with 2 units, the winner-vs-loser difference mode has
    # eigenvalue == gating_self_weight exactly, so >1 here makes it a growing
    # map that saturates and then holds at the clipped ceiling on its own,
    # rather than decaying back toward baseline once the driving cue turns
    # off. This is what lets a brief cue flash leave one gate latched on for
    # the rest of the trial -- a bistable, self-sustaining winner -- instead
    # of needing the cue itself to stay on that whole time.
    gating_self_weight: float = 1.6
    gating_lateral_weight: float = -0.45

    conjunction_noise_standard_deviation: float = 0.005

    fast_learning_rate: float = 0.02
    slow_learning_rate: float = 0.0002
    maximum_fast_weight: float = 1.0
    maximum_slow_weight: float = 0.2

    # blend of the two timescales into the effective weight W (old
    # w_short_weight / w_long_weight). 1.0 each = plain sum, as published.
    fast_weight_blend: float = 1.0
    slow_weight_blend: float = 1.0

    def __post_init__(self) -> None:
        if self.number_of_conjunction_units <= 0:
            raise ValueError('number_of_conjunction_units must be positive')
        if self.num_cues_per_rule <= 0:
            raise ValueError('num_cues_per_rule must be positive')
        if not 0.0 <= self.baseline_activity <= 1.0:
            raise ValueError('baseline_activity must lie between 0 and 1')
        if self.conjunction_noise_standard_deviation < 0:
            raise ValueError('Noise standard deviation cannot be negative')
        if self.fast_learning_rate < 0 or self.slow_learning_rate < 0:
            raise ValueError('Learning rates cannot be negative')
        if self.maximum_fast_weight <= 0 or self.maximum_slow_weight <= 0:
            raise ValueError('Maximum weights must be positive')
        if self.number_of_gating_units not in (0, 2):
            raise ValueError('number_of_gating_units must be 0 or 2 (one gate per rule)')
        if self.cue_to_gating_gain < 0 or self.gating_to_feature_gain < 0:
            raise ValueError('Gating gains cannot be negative')
        if self.gating_to_relevant_feature_gain < 0:
            raise ValueError('gating_to_relevant_feature_gain cannot be negative')
        if self.gating_fast_learning_rate < 0 or self.gating_slow_learning_rate < 0:
            raise ValueError('Gating learning rates cannot be negative')
        if self.gating_maximum_fast_weight <= 0 or self.gating_maximum_slow_weight <= 0:
            raise ValueError('Gating maximum weights must be positive')
        if not 0.0 <= self.gating_trace_decay <= 1.0:
            raise ValueError('gating_trace_decay must lie between 0 and 1')


@dataclass(frozen=True)
class NetworkState:
    """Activity of all neural populations after one update.

    gating_activity is length-0 unless gating units are enabled.
    """

    feature_activity: np.ndarray
    conjunction_activity: np.ndarray
    gating_activity: np.ndarray


def _bounded_activity(values: np.ndarray) -> np.ndarray:
    """Keep every unit between silence (0) and maximum activity (1)."""

    return np.clip(values, 0.0, 1.0)


class PlasticAttractor:
    """Feature units recurrently coupled to conjunction units.

    Feature-vector size follows parameters.num_cues_per_rule (10 units, the
    default, at num_cues_per_rule=2); conjunction population size follows
    parameters.number_of_conjunction_units.
    """

    def __init__(
        self,
        seed: int = 0,
        parameters: ModelParameters | None = None,
        learn: bool = True,
        *,
        random_generator: np.random.RandomState | None = None,
    ) -> None:
        self.parameters = parameters or ModelParameters()
        self.learn = learn

        vocabulary = build_vocabulary(self.parameters.num_cues_per_rule)
        self._number_of_feature_units = vocabulary.number_of_features
        self._competing_feature_groups = vocabulary.competing_feature_groups

        # Cue units feed the gates; only colour/shape units may be inhibited.
        self._cue_indices = np.array(vocabulary.all_cue_indices)
        self._suppressible_indices = np.array(vocabulary.suppressible_feature_indices)
        self._gate_target_indices = vocabulary.gate_target_indices
        self._gate_relevant_target_indices = vocabulary.gate_relevant_target_indices

        # The experiment can provide one shared generator so initialization,
        # trial order, and neural noise all remain reproducible from one seed.
        self._random = (
            random_generator
            if random_generator is not None
            else np.random.RandomState(seed)
        )

        self._feature_recurrent_weights = self._build_feature_connections()
        self._conjunction_recurrent_weights = self._build_conjunction_connections()
        self._initialize_plastic_weights()

        # Gate weights draw from the RNG only when gating is on, so the stream
        # (and every published weight) stays identical when it is off.
        if self.number_of_gating_units > 0:
            self._gating_recurrent_weights = self._build_gating_connections()
            self._initialize_gating_weights()
            self._gating_relevant_mask = self._build_gating_relevant_mask()

        # Every instruction or trial epoch starts from zero activity.
        self._feature_activity = np.zeros(self.number_of_feature_units)
        self._conjunction_activity = np.zeros(self.number_of_conjunction_units)
        self._gating_activity = np.zeros(self.number_of_gating_units)
        self._reset_gating_trace()

    @property
    def number_of_feature_units(self) -> int:
        return self._number_of_feature_units

    @property
    def number_of_conjunction_units(self) -> int:
        return self.parameters.number_of_conjunction_units

    @property
    def number_of_gating_units(self) -> int:
        return self.parameters.number_of_gating_units

    @property
    def state(self) -> NetworkState:
        """Return copies so callers cannot alter the live activity."""

        return NetworkState(
            feature_activity=self._feature_activity.copy(),
            conjunction_activity=self._conjunction_activity.copy(),
            gating_activity=self._gating_activity.copy(),
        )

    @property
    def combined_weights(self) -> np.ndarray:
        """Return W, the blended sum of the fast and slow weight matrices."""

        return self._combined_weights.copy()

    @property
    def fast_weights(self) -> np.ndarray:
        return self._fast_weights.copy()

    @property
    def slow_weights(self) -> np.ndarray:
        return self._slow_weights.copy()

    @property
    def gating_input_weights(self) -> np.ndarray:
        """Plastic cue -> gate weights, shape (num_gates, num_cues)."""

        return self._gating_input_weights.copy()

    @property
    def gating_output_weights(self) -> np.ndarray:
        """Plastic gate -> feature magnitudes, shape (num_features, num_gates).

        Nonzero only on the colour/shape rows; the inhibitory sign is applied in
        the feature update, not stored here.
        """

        return self._gating_output_weights.copy()

    def reset_activity(self) -> None:
        """Reset activity while preserving everything the weights learned.

        Also clears the gate eligibility trace: it must not bleed across
        trials, exactly like activity itself.
        """

        self._feature_activity = np.zeros(self.number_of_feature_units)
        self._conjunction_activity = np.zeros(self.number_of_conjunction_units)
        self._gating_activity = np.zeros(self.number_of_gating_units)
        self._reset_gating_trace()

    def _reset_gating_trace(self) -> None:
        self._gating_input_trace = np.zeros(
            (self.number_of_gating_units, self._cue_indices.size)
        )
        self._gating_output_trace = np.zeros(
            (self._suppressible_indices.size, self.number_of_gating_units)
        )

    def consolidate_gating_trace(self, reward: float) -> None:
        """Apply the trial's accumulated gate eligibility trace, reward-gated.

        A three-factor rule: coincident cue/gate/feature activity across the
        trial builds an eligibility trace (see step()), but it is not applied
        to the weights until the outcome is known. reward > 0 (correct trial)
        potentiates as before; reward < 0 (incorrect trial) applies the same
        trace with a flipped sign, i.e. depression -- the bidirectional
        dopamine-gated LTP/LTD pattern reported at corticostriatal synapses
        (Shen et al. 2008). Has no effect when gating is disabled.
        """

        if self.number_of_gating_units == 0:
            return

        parameters = self.parameters
        self._gating_input_fast_weights = np.clip(
            self._gating_input_fast_weights
            + reward * parameters.gating_fast_learning_rate * self._gating_input_trace,
            0.0,
            parameters.gating_maximum_fast_weight,
        )
        self._gating_input_slow_weights = np.clip(
            self._gating_input_slow_weights
            + reward * parameters.gating_slow_learning_rate * self._gating_input_trace,
            0.0,
            parameters.gating_maximum_slow_weight,
        )

        rows = self._suppressible_indices
        self._gating_output_fast_weights[rows] = np.clip(
            self._gating_output_fast_weights[rows]
            + reward * parameters.gating_fast_learning_rate * self._gating_output_trace,
            0.0,
            parameters.gating_maximum_fast_weight,
        )
        self._gating_output_slow_weights[rows] = np.clip(
            self._gating_output_slow_weights[rows]
            + reward * parameters.gating_slow_learning_rate * self._gating_output_trace,
            0.0,
            parameters.gating_maximum_slow_weight,
        )

        self._combine_gating_weights()
        self._reset_gating_trace()

    def step(
        self,
        external_input: np.ndarray,
        *,
        perturbation: object | None = None,
        perturbation_active: bool = False,
        gate_external_input: np.ndarray | None = None,
        cue_signal_active: bool = False,
        pause_weight_learning: bool = False,
        learn: bool | None = None,
    ) -> NetworkState:
        """Advance the neural and Hebbian equations by one time step."""

        input_vector = np.asarray(external_input, dtype=float)
        expected_shape = (self.number_of_feature_units,)
        if input_vector.shape != expected_shape:
            raise ValueError(
                f'external_input must have shape {expected_shape}; '
                f'received {input_vector.shape}'
            )

        parameters = self.parameters

        # The equations measure activity relative to the resting baseline.
        centered_features = (
            self._feature_activity - parameters.baseline_activity
        )
        centered_conjunctions = (
            self._conjunction_activity - parameters.baseline_activity
        )

        if perturbation is not None and perturbation_active:
            clamp_value = float(getattr(perturbation, 'clamp_value', 0.0))
            if getattr(perturbation, 'target_conjunction_units', False):
                self._conjunction_activity.fill(clamp_value)
                centered_conjunctions = (
                    self._conjunction_activity - parameters.baseline_activity
                )
            if getattr(perturbation, 'target_gating_units', False):
                self._gating_activity.fill(clamp_value)

        # Gate inhibition of the feature update reads the previous gate activity,
        # so all populations update synchronously from the same prior state. It
        # is exactly zero (and touches no gate weights) when gating is off, so
        # the feature update stays bit-identical to the published model.
        gating_inhibition = (
            parameters.gating_to_feature_gain
            * (
                self._gating_output_weights
                @ (self._gating_activity - parameters.baseline_activity)
            )
            if self.number_of_gating_units > 0
            else 0.0
        )

        # Fixed (not learned) excitatory counterpart of gating_inhibition: a
        # winning gate boosts its OWN dimension's rows (see
        # gating_to_relevant_feature_gain), rather than only suppressing the
        # other one. Multiplicative in the feature's OWN current deviation
        # from baseline (centered_features below), not a flat additive
        # constant: a flat constant was tried first and applies equally to
        # BOTH members of the relevant pair (e.g. green and blue alike, since
        # the gate only knows the relevant DIMENSION, not which member is
        # actually presented) -- that symmetrically inflates the *absent*
        # member too, collapsing the within-pair discrimination it was meant
        # to sharpen (confirmed empirically: conjunction_unit_discrimination's
        # well-separated-unit count fell to ~0 at every enhancement gain > 0
        # tried, regardless of inhibition strength). Multiplying by the
        # feature's own centered activity instead amplifies whichever member
        # is already driven up by the actual stimulus (near-zero deviation on
        # the absent member gives near-zero enhancement), so it sharpens the
        # existing presented-vs-absent contrast rather than blurring it. Zero
        # whenever the gain is left at its default 0.0, so this stays a
        # no-op unless explicitly enabled.
        gating_enhancement = (
            parameters.gating_to_relevant_feature_gain
            * (
                self._gating_relevant_mask
                @ (self._gating_activity - parameters.baseline_activity)
            )
            * centered_features
            if self.number_of_gating_units > 0
            else 0.0
        )

        # Feature units receive recurrent input, top-down input through W, the
        # stimulus/cue/teaching supplied by the experiment for this step, and
        # (when enabled) subtractive inhibition and multiplicative enhancement
        # from the gates.
        next_features = _bounded_activity(
            parameters.baseline_activity
            + self._feature_recurrent_weights @ centered_features
            + parameters.conjunction_to_feature_gain
            * (self._combined_weights @ centered_conjunctions)
            + input_vector
            - gating_inhibition
            + gating_enhancement
        )

        # Conjunction units receive recurrent input, bottom-up input through W
        # transposed, and a small amount of random neural variation.
        next_conjunctions = _bounded_activity(
            parameters.baseline_activity
            + self._conjunction_recurrent_weights @ centered_conjunctions
            + parameters.feature_to_conjunction_gain
            * (self._combined_weights.T @ centered_features)
            + parameters.conjunction_noise_standard_deviation
            * self._random.standard_normal(self.number_of_conjunction_units)
        )

        # Gate units are feedforward: driven by the previous cue activity plus
        # any teaching drive the experiment supplies this step, but only while
        # cue_signal_active -- see _next_gating_activity for why.
        next_gates = self._next_gating_activity(gate_external_input, cue_signal_active)

        if perturbation is not None and perturbation_active:
            clamp_value = float(getattr(perturbation, 'clamp_value', 0.0))
            if getattr(perturbation, 'target_conjunction_units', False):
                next_conjunctions.fill(clamp_value)
            if getattr(perturbation, 'target_gating_units', False):
                next_gates.fill(clamp_value)

        # All populations are updated synchronously from the previous state.
        self._feature_activity = next_features
        self._conjunction_activity = next_conjunctions
        self._gating_activity = next_gates

        learning_is_on = self.learn if learn is None else learn
        if learning_is_on:
            # pause_weight_learning only excludes the currently-winning
            # gate's target rows (see _update_plastic_weights); the relevant
            # dimension, action/cue rows, and the gate's own learning below
            # are unaffected. The caller only ever sets it when gating is
            # on, so with gating off every row updates every step exactly
            # as in the published model.
            self._update_plastic_weights(pause_weight_learning)
            # The trace only forms while the caller marks this step as inside
            # the cue's genuine presentation window (cue_signal_active; the
            # experiment ties this to stimulus_window, not to whether a forced
            # gate_external_input happens to be set). This is what makes
            # accumulation safe on real trials, which supply no forced drive at
            # all: without this check, or with a window longer than the cue is
            # actually driven, the trace would keep accumulating after the cue
            # decays back toward baseline, and two units both silently below
            # baseline produce a spurious *positive* covariance term (see
            # hebbian-window-baseline-centering-pitfall).
            if self.number_of_gating_units > 0 and cue_signal_active:
                self._accumulate_gating_trace()

        return self.state

    def _next_gating_activity(
        self, gate_external_input: np.ndarray | None, cue_signal_active: bool
    ) -> np.ndarray:
        """Feedforward gate update: self-sustaining recurrence + a cue input
        that only counts while cue_signal_active.

        Reading cue-unit activity every step (rather than only while
        cue_signal_active) let the gate get dragged off its own, already-
        settled decision: cue units aren't just externally driven and then
        silent, they get *re-excited* by bottom-up conjunction feedback as
        the trial settles into whichever attractor the main network favours,
        and after a run of same-rule trials that attractor can be strong
        enough to partially re-light the *wrong* cue mid-trial. The gate is
        meant to hold its decision via its own recurrence (gating_self_weight)
        once the genuine cue is gone, not keep re-deriving it from a channel
        that's become entangled with the network's own settling dynamics.
        """

        if self.number_of_gating_units == 0:
            return self._gating_activity

        parameters = self.parameters
        if gate_external_input is None:
            gate_drive = np.zeros(self.number_of_gating_units)
        else:
            gate_drive = np.asarray(gate_external_input, dtype=float)
            if gate_drive.shape != (self.number_of_gating_units,):
                raise ValueError(
                    f'gate_external_input must have shape '
                    f'{(self.number_of_gating_units,)}; received {gate_drive.shape}'
                )

        cue_term = 0.0
        if cue_signal_active:
            centered_cues = (
                self._feature_activity[self._cue_indices] - parameters.baseline_activity
            )
            cue_term = parameters.cue_to_gating_gain * (
                self._gating_input_weights @ centered_cues
            )

        return _bounded_activity(
            parameters.baseline_activity
            + self._gating_recurrent_weights
            @ (self._gating_activity - parameters.baseline_activity)
            + cue_term
            + gate_drive
        )

    def feedback_eigenvalues(self) -> np.ndarray:
        """Return the eigenvalues of W W.T used in the paper's analysis.

        Length equals the number of feature units (one per row of W W.T), so
        most come out near zero because rank(W W.T) is capped at the number of
        conjunction units.
        """

        return np.linalg.eigvalsh(
            self._combined_weights @ self._combined_weights.T
        )

    def number_of_amplifying_eigenvalues(self) -> int:
        """Count feedback eigenvalues greater than one."""

        return int(np.sum(self.feedback_eigenvalues() > 1.0))

    def _build_feature_connections(self) -> np.ndarray:
        # block-diagonal lateral inhibition: a 1 wherever two units share a
        # feature group, so competition stays local to each group. np.ix_
        # handles any group size, including the four-unit cue group.
        #
        # for the default 10-unit vocabulary (num_cues_per_rule=2) the four
        # groups are (0,1) color, (2,3) shape, (4,5,6,7) cues, (8,9) action,
        # so within_dimension is a 10x10 matrix with 1s in four square blocks
        # on the diagonal (including each unit's own cell) and 0 elsewhere:
        #   [[1,1,0,0,0,0,0,0,0,0],
        #    [1,1,0,0,0,0,0,0,0,0],
        #    [0,0,1,1,0,0,0,0,0,0],
        #    [0,0,1,1,0,0,0,0,0,0],
        #    [0,0,0,0,1,1,1,1,0,0],
        #    [0,0,0,0,1,1,1,1,0,0],
        #    [0,0,0,0,1,1,1,1,0,0],
        #    [0,0,0,0,1,1,1,1,0,0],
        #    [0,0,0,0,0,0,0,0,1,1],
        #    [0,0,0,0,0,0,0,0,1,1]]
        within_dimension = np.zeros(
            (self.number_of_feature_units, self.number_of_feature_units)
        )
        for group in self._competing_feature_groups:
            within_dimension[np.ix_(group, group)] = 1.0

        # since within_dimension is already 1 on the diagonal, the returned
        # matrix's diagonal is self_weight + lateral_weight (0.73 - 0.28 =
        # 0.45), its within-group off-diagonal entries are lateral_weight
        # alone (-0.28), and entries between different groups stay 0 (no
        # recurrent interaction across groups).
        parameters = self.parameters
        return (
            parameters.feature_self_weight * np.eye(self.number_of_feature_units)
            + parameters.feature_lateral_weight * within_dimension
        )

    def _build_conjunction_connections(self) -> np.ndarray:
        # global lateral inhibition: every conjunction unit inhibits every
        # other equally, so competition is winner-take-all across the population.
        parameters = self.parameters
        return (
            parameters.conjunction_self_weight
            * np.eye(self.number_of_conjunction_units)
            + parameters.conjunction_lateral_weight
            * np.ones(
                (self.number_of_conjunction_units, self.number_of_conjunction_units)
            )
        )

    def _build_gating_connections(self) -> np.ndarray:
        # global lateral inhibition between gates, exactly like the conjunction
        # units: gating_self_weight on the diagonal plus gating_lateral_weight
        # everywhere, so the gate the cue drives hardest wins and suppresses the
        # other. Built only when gating is enabled.
        parameters = self.parameters
        number_of_gates = self.number_of_gating_units
        return (
            parameters.gating_self_weight * np.eye(number_of_gates)
            + parameters.gating_lateral_weight
            * np.ones((number_of_gates, number_of_gates))
        )

    def _build_gating_relevant_mask(self) -> np.ndarray:
        # fixed (not learned) 0/1 map, shape (num_features, num_gates): 1 at
        # (feature_index, gate_index) wherever that gate's OWN dimension
        # includes that feature (task.py's gate_relevant_target_indices) --
        # the mirror image of _gating_output_mask's irrelevant-dimension
        # targets. Used only to scale gating_to_relevant_feature_gain in
        # step(); never touched by any learning rule.
        mask = np.zeros((self.number_of_feature_units, self.number_of_gating_units))
        for gate_index, target_indices in enumerate(self._gate_relevant_target_indices):
            for feature_index in target_indices:
                mask[feature_index, gate_index] = 1.0
        return mask

    def _initialize_plastic_weights(self) -> None:
        # shape is (num_features, num_conjunction) = (10, 4): W drives the
        # top-down feature update as-is and the bottom-up conjunction update as
        # W.T, so the same synapses are read in whichever direction is needed.
        shape = (self.number_of_feature_units, self.number_of_conjunction_units)
        parameters = self.parameters

        self._fast_weights = self._random.uniform(
            0.0,
            parameters.maximum_fast_weight,
            size=shape,
        )
        self._slow_weights = self._random.uniform(
            0.0,
            parameters.maximum_slow_weight,
            size=shape,
        )
        # Cues reach the network only through the gating pathway (see
        # cue_to_gating_gain / gating_to_feature_gain), never through the
        # colour/shape/action conjunctive code, so their W rows start at zero.
        self._fast_weights[self._cue_indices] = 0.0
        self._slow_weights[self._cue_indices] = 0.0
        self._combine_weights()

    def _initialize_gating_weights(self) -> None:
        # cue -> gate (num_gates, num_cues): plastic excitatory input, learns
        # which cue drives which gate. gate -> feature (num_features, num_gates):
        # plastic magnitude, nonzero only on the colour/shape rows the gate may
        # inhibit, so it can never suppress cues or actions. Each is split into
        # a fast and slow component, like the conjunction<->feature weights,
        # so the slow component anchors what instruction taught against the
        # unsupervised drift real-block trials would otherwise cause.
        parameters = self.parameters
        num_gates = self.number_of_gating_units
        input_shape = (num_gates, self._cue_indices.size)
        output_shape = (self._suppressible_indices.size, num_gates)

        self._gating_input_fast_weights = self._random.uniform(
            0.0, parameters.gating_maximum_fast_weight, size=input_shape,
        )
        self._gating_input_slow_weights = self._random.uniform(
            0.0, parameters.gating_maximum_slow_weight, size=input_shape,
        )

        # each gate may only ever land weight on the OTHER dimension's rows
        # (colour gate -> shape rows, shape gate -> colour rows), hardcoded
        # from gate_target_indices rather than left for the reward-gated
        # trace to discover -- both dimensions are simultaneously active with
        # the correct gate throughout every stimulus_window regardless of
        # which one the rule cares about, so an unconstrained trace
        # potentiates suppression of a gate's own dimension just as readily
        # as the genuinely irrelevant one.
        row_position = {
            feature_index: position
            for position, feature_index in enumerate(self._suppressible_indices)
        }
        self._gating_output_mask = np.zeros((self._suppressible_indices.size, num_gates), dtype=bool)
        for gate_index, target_indices in enumerate(self._gate_target_indices):
            for feature_index in target_indices:
                self._gating_output_mask[row_position[feature_index], gate_index] = True

        self._gating_output_fast_weights = np.zeros(
            (self.number_of_feature_units, num_gates)
        )
        self._gating_output_fast_weights[self._suppressible_indices] = (
            self._random.uniform(0.0, parameters.gating_maximum_fast_weight, size=output_shape)
            * self._gating_output_mask
        )
        self._gating_output_slow_weights = np.zeros(
            (self.number_of_feature_units, num_gates)
        )
        self._gating_output_slow_weights[self._suppressible_indices] = (
            self._random.uniform(0.0, parameters.gating_maximum_slow_weight, size=output_shape)
            * self._gating_output_mask
        )

        self._combine_gating_weights()

    def _update_plastic_weights(self, pause_weight_learning: bool = False) -> None:
        parameters = self.parameters

        # The outer product gives one covariance-style Hebbian change per
        # connection: units on the same side of baseline strengthen, opposite
        # sides weaken.
        change = np.outer(
            self._feature_activity - parameters.baseline_activity,
            self._conjunction_activity - parameters.baseline_activity,
        )
        # Cue rows never learn: same restriction as init, applied every step
        # so Hebbian drift can't move them off zero.
        change[self._cue_indices] = 0.0
        # Only the rows the CURRENTLY-WINNING gate targets, and only during
        # the caller's plasticity-pause window (see
        # EpochProtocol.plasticity_pause_buffer_steps) -- not the whole
        # colour/shape block. gate_target_indices already says which rows
        # each gate may ever suppress (its task's irrelevant pair, never
        # its own -- see task.py); reusing that same mask here excludes
        # exactly the dimension this trial's winning gate is trying to
        # suppress, while leaving the RELEVANT dimension's rows (and
        # action/cue rows, which were never suppressible in the first
        # place) learning every step. An earlier version of this pause
        # excluded the entire suppressible block regardless of which gate
        # was winning -- on every trial that also zeroed the relevant
        # dimension's stimulus_window contribution for no reason, and
        # measurably hurt accuracy instead of helping it.
        if pause_weight_learning and self.number_of_gating_units > 0:
            winning_gate = int(np.argmax(self._gating_activity))
            irrelevant_rows = self._suppressible_indices[
                self._gating_output_mask[:, winning_gate]
            ]
            change[irrelevant_rows] = 0.0

        # fast weights swing large and bound high; slow weights drift ~100x
        # slower and bound low.
        self._fast_weights = np.clip(
            self._fast_weights + parameters.fast_learning_rate * change,
            0.0,
            parameters.maximum_fast_weight,
        )
        self._slow_weights = np.clip(
            self._slow_weights + parameters.slow_learning_rate * change,
            0.0,
            parameters.maximum_slow_weight,
        )
        self._combine_weights()

    def _accumulate_gating_trace(self) -> None:
        # Three-factor rule (Fremaux & Gerstner 2016): coincident cue/gate/
        # feature activity forms an eligibility trace every step, exactly the
        # raw Hebbian outer product used elsewhere, but it is NOT applied to
        # the weights here. The trial's dynamics run under fixed gate weights
        # throughout; consolidate_gating_trace() applies the accumulated trace,
        # scaled by trial outcome, once the response is known.
        parameters = self.parameters
        centered_features = self._feature_activity - parameters.baseline_activity
        centered_cues = (
            self._feature_activity[self._cue_indices] - parameters.baseline_activity
        )
        centered_gates = self._gating_activity - parameters.baseline_activity
        rows = self._suppressible_indices

        decay = parameters.gating_trace_decay
        self._gating_input_trace = (
            decay * self._gating_input_trace
            + np.outer(centered_gates, centered_cues)
        )
        # masked to gate_target_indices so a gate's own dimension can never
        # accumulate a nonzero trace, matching how _update_plastic_weights
        # masks change[cue_indices] every step rather than only at init.
        output_contribution = np.outer(centered_features, centered_gates)[rows]
        output_contribution[~self._gating_output_mask] = 0.0
        self._gating_output_trace = (
            decay * self._gating_output_trace
            + output_contribution
        )

    def _combine_gating_weights(self) -> None:
        parameters = self.parameters
        self._gating_input_weights = (
            parameters.gating_fast_weight_blend * self._gating_input_fast_weights
            + parameters.gating_slow_weight_blend * self._gating_input_slow_weights
        )
        self._gating_output_weights = (
            parameters.gating_fast_weight_blend * self._gating_output_fast_weights
            + parameters.gating_slow_weight_blend * self._gating_output_slow_weights
        )

    def _combine_weights(self) -> None:
        # W = fast_blend*fast + slow_blend*slow (both blends 1.0 by default).
        parameters = self.parameters
        self._combined_weights = (
            parameters.fast_weight_blend * self._fast_weights
            + parameters.slow_weight_blend * self._slow_weights
        )
