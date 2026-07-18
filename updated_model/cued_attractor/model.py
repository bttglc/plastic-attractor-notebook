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


@dataclass(frozen=True)
class NetworkState:
    """Activity of both neural populations after one update."""

    feature_activity: np.ndarray
    conjunction_activity: np.ndarray


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

        # Every instruction or trial epoch starts from zero activity.
        self._feature_activity = np.zeros(self.number_of_feature_units)
        self._conjunction_activity = np.zeros(self.number_of_conjunction_units)

    @property
    def number_of_feature_units(self) -> int:
        return self._number_of_feature_units

    @property
    def number_of_conjunction_units(self) -> int:
        return self.parameters.number_of_conjunction_units

    @property
    def state(self) -> NetworkState:
        """Return copies so callers cannot alter the live activity."""

        return NetworkState(
            feature_activity=self._feature_activity.copy(),
            conjunction_activity=self._conjunction_activity.copy(),
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

    def reset_activity(self) -> None:
        """Reset activity while preserving everything the weights learned."""

        self._feature_activity = np.zeros(self.number_of_feature_units)
        self._conjunction_activity = np.zeros(self.number_of_conjunction_units)

    def step(
        self,
        external_input: np.ndarray,
        *,
        clamp_conjunctions: bool = False,
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

        # Feature units receive recurrent input, top-down input through W, and
        # the stimulus/cue/teaching supplied by the experiment for this step.
        next_features = _bounded_activity(
            parameters.baseline_activity
            + self._feature_recurrent_weights @ centered_features
            + parameters.conjunction_to_feature_gain
            * (self._combined_weights @ centered_conjunctions)
            + input_vector
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

        # The paper's TMS-like manipulation forces all conjunction units to 1.
        if clamp_conjunctions:
            next_conjunctions.fill(1.0)

        # Both populations are updated synchronously from the previous state.
        self._feature_activity = next_features
        self._conjunction_activity = next_conjunctions

        learning_is_on = self.learn if learn is None else learn
        if learning_is_on:
            self._update_plastic_weights()

        return self.state

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
        within_dimension = np.zeros(
            (self.number_of_feature_units, self.number_of_feature_units)
        )
        for group in self._competing_feature_groups:
            within_dimension[np.ix_(group, group)] = 1.0

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
        self._combine_weights()

    def _update_plastic_weights(self) -> None:
        parameters = self.parameters

        # The outer product gives one covariance-style Hebbian change per
        # connection: units on the same side of baseline strengthen, opposite
        # sides weaken.
        change = np.outer(
            self._feature_activity - parameters.baseline_activity,
            self._conjunction_activity - parameters.baseline_activity,
        )

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

    def _combine_weights(self) -> None:
        # W = fast_blend*fast + slow_blend*slow (both blends 1.0 by default).
        parameters = self.parameters
        self._combined_weights = (
            parameters.fast_weight_blend * self._fast_weights
            + parameters.slow_weight_blend * self._slow_weights
        )
