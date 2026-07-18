import os
import sys
import unittest

import numpy as np

# make the sibling cued_attractor package importable regardless of the cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cued_attractor import (
    SwitchingExperimentConfig,
    Task,
    TransitionType,
    run_switching_experiment,
    summarize_behavior,
)


def _small_config(seed: int = 0) -> SwitchingExperimentConfig:
    # 2 instruction blocks x 8 trials + 2 performance-practice blocks x 4
    # trials + 1 real block x 4 trials = 28 trials
    return SwitchingExperimentConfig(
        seed=seed,
        num_practice_blocks=2,
        practice_permutation_repeats=1,
        num_trials=4,
        switch_probs=(0.5,),
    )


class CuedSwitchingExperimentTests(unittest.TestCase):
    def test_block_structure_and_trial_counts(self) -> None:
        result = run_switching_experiment(_small_config())

        self.assertEqual(len(result.trials), 28)
        # five blocks: two instruction, two performance-practice, one real
        self.assertEqual(len(result.amplifying_eigenvalue_count_by_block), 5)
        self.assertEqual(result.final_combined_weights.shape, (10, 4))

    def test_practice_blocks_come_first_and_teach_one_rule_each(self) -> None:
        result = run_switching_experiment(_small_config())

        practice = [trial for trial in result.trials if trial.is_practice]
        real = [trial for trial in result.trials if not trial.is_practice]

        # practice precedes real, and each practice block teaches a single rule
        self.assertEqual(len(practice), 24)
        self.assertEqual(len(real), 4)

        instruction = [trial for trial in practice if trial.is_instruction]
        performance_practice = [trial for trial in practice if not trial.is_instruction]
        self.assertEqual(len(instruction), 16)
        self.assertEqual(len(performance_practice), 8)

        self.assertTrue(all(trial.task == Task.COLOR for trial in instruction[:8]))
        self.assertTrue(all(trial.task == Task.SHAPE for trial in instruction[8:]))
        self.assertTrue(all(trial.task == Task.COLOR for trial in performance_practice[:4]))
        self.assertTrue(all(trial.task == Task.SHAPE for trial in performance_practice[4:]))

        # a constant-rule practice block never contains a rule switch
        self.assertTrue(
            all(
                trial.transition_type != TransitionType.RULE_SWITCH
                for trial in practice
            )
        )

    def test_seeded_experiments_are_reproducible(self) -> None:
        first = run_switching_experiment(_small_config(seed=3))
        second = run_switching_experiment(_small_config(seed=3))

        self.assertEqual(
            [trial.chosen_response for trial in first.trials],
            [trial.chosen_response for trial in second.trials],
        )
        np.testing.assert_allclose(
            first.final_combined_weights,
            second.final_combined_weights,
        )

    def test_practice_teaches_above_chance_behaviour(self) -> None:
        # a single moderate run: practice should push real-block accuracy above
        # chance and leave at least one amplifying eigenvalue. behavioural
        # magnitudes are only trustworthy at ~20 seeds, so this asserts a floor
        # rather than the full published signature
        config = SwitchingExperimentConfig(
            seed=0,
            num_practice_blocks=2,
            practice_permutation_repeats=6,
            num_trials=24,
            switch_probs=(0.5,),
        )
        result = run_switching_experiment(config)
        real_trials = [t for t in result.trials if not t.is_practice]
        summary = summarize_behavior(real_trials)

        self.assertGreater(summary.accuracy, 0.5)
        self.assertGreaterEqual(
            result.amplifying_eigenvalue_count_by_block[-1],
            1,
        )


if __name__ == '__main__':
    unittest.main()
