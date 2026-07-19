import os
import sys
import unittest

import numpy as np

# make the gated_simple_attractor package importable regardless of the cwd
# (this file sits one level deeper than the flat tests, inside the package folder)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gated_simple_attractor import ModelParameters, PlasticAttractor


class PlasticAttractorTests(unittest.TestCase):
    def test_default_network_has_the_expected_dimensions(self) -> None:
        model = PlasticAttractor(seed=0)

        # six feature units (no cues), four conjunction units
        self.assertEqual(model.number_of_feature_units, 6)
        self.assertEqual(model.number_of_conjunction_units, 4)
        self.assertEqual(model.state.feature_activity.shape, (6,))
        self.assertEqual(model.state.conjunction_activity.shape, (4,))
        self.assertEqual(model.combined_weights.shape, (6, 4))

    def test_combined_weights_are_the_blended_fast_and_slow_weights(self) -> None:
        model = PlasticAttractor(seed=0)

        # default blends are 1.0, so W is the plain sum
        np.testing.assert_allclose(
            model.combined_weights,
            model.fast_weights + model.slow_weights,
        )

    def test_weight_blend_scales_each_timescale(self) -> None:
        parameters = ModelParameters(fast_weight_blend=2.0, slow_weight_blend=0.5)
        model = PlasticAttractor(seed=0, parameters=parameters)

        np.testing.assert_allclose(
            model.combined_weights,
            2.0 * model.fast_weights + 0.5 * model.slow_weights,
        )

    def test_default_parameters_use_the_public_repository_gains(self) -> None:
        parameters = ModelParameters()

        self.assertEqual(parameters.conjunction_to_feature_gain, 0.08)
        self.assertEqual(parameters.feature_to_conjunction_gain, 0.04)

    def test_one_step_is_reproducible_and_keeps_activity_bounded(self) -> None:
        first_model = PlasticAttractor(seed=7)
        second_model = PlasticAttractor(seed=7)
        # a six-long input: green + circle present, everything else silent
        external_input = np.zeros(6)
        external_input[[0, 3]] = 1.0

        first_state = first_model.step(external_input)
        second_state = second_model.step(external_input)

        np.testing.assert_allclose(
            first_state.feature_activity,
            second_state.feature_activity,
        )
        np.testing.assert_allclose(
            first_state.conjunction_activity,
            second_state.conjunction_activity,
        )
        self.assertTrue(np.all(0.0 <= first_state.feature_activity))
        self.assertTrue(np.all(first_state.feature_activity <= 1.0))
        self.assertTrue(np.all(0.0 <= first_state.conjunction_activity))
        self.assertTrue(np.all(first_state.conjunction_activity <= 1.0))


if __name__ == '__main__':
    unittest.main()
