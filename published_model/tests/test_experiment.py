import unittest

import numpy as np

from plastic_attractor import (
    BlockedExperimentConfig,
    Feature,
    Task,
    run_baseline,
    run_blocked_experiment,
    summarize_behavior,
)


class BlockedExperimentTests(unittest.TestCase):
    def test_beginner_entry_point_needs_only_plain_arguments(self) -> None:
        result = run_baseline(seed=0, number_of_blocks=2)

        self.assertEqual(len(result.trials), 24)

    def test_two_blocks_form_a_small_complete_experiment(self) -> None:
        result = run_blocked_experiment(
            BlockedExperimentConfig(seed=0, number_of_blocks=2)
        )

        self.assertEqual(len(result.trials), 24)
        self.assertEqual(result.trials[0].task, Task.COLOR)
        self.assertEqual(result.trials[12].task, Task.SHAPE)
        self.assertEqual(result.final_combined_weights.shape, (6, 4))
        self.assertEqual(len(result.amplifying_eigenvalue_count_by_block), 2)
        self.assertIn(
            result.trials[0].correct_response,
            (Feature.LEFT, Feature.RIGHT),
        )

    def test_seeded_experiments_are_reproducible(self) -> None:
        config = BlockedExperimentConfig(seed=3, number_of_blocks=2)

        first = run_blocked_experiment(config)
        second = run_blocked_experiment(config)

        self.assertEqual(
            [trial.chosen_response for trial in first.trials],
            [trial.chosen_response for trial in second.trials],
        )
        np.testing.assert_allclose(
            first.final_combined_weights,
            second.final_combined_weights,
        )

    def test_published_design_retains_its_main_behavioral_signature(self) -> None:
        result = run_blocked_experiment(
            BlockedExperimentConfig(seed=0, number_of_blocks=20)
        )
        summary = summarize_behavior(result.trials)

        self.assertEqual(len(result.trials), 240)
        self.assertGreater(summary.accuracy, 0.90)
        self.assertGreater(summary.congruency_effect_in_steps, 0.0)
        self.assertGreaterEqual(
            result.amplifying_eigenvalue_count_by_block[-1],
            1,
        )


if __name__ == "__main__":
    unittest.main()
