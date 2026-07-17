"""Run the published blocked experiment and print its main results.

Start here if you want to use the multi-file model before reading its classes.
Run this file from the repository's top-level folder with:

    python3 examples/run_baseline.py
"""

from plastic_attractor import run_baseline, summarize_behavior


# A seed makes the random initialization and neural noise reproducible.
# Twenty blocks reproduce the full baseline design: 12 trials in every block.
result = run_baseline(seed=0, number_of_blocks=20)

# The simulation keeps every trial. This function reduces those trial-level
# records to the main behavioral measurements used in the notebook.
summary = summarize_behavior(result.trials)

print("Plastic-attractor blocked baseline")
print(f"Trials: {len(result.trials)}")
print(f"Accuracy: {summary.accuracy:.1%}")
print(
    "Mean reaction time: "
    f"{summary.mean_reaction_time_in_steps:.2f} model steps"
)
print(
    "Congruency effect: "
    f"{summary.congruency_effect_in_steps:.2f} model steps"
)
print(
    "Amplifying eigenvalues after the final block: "
    f"{result.amplifying_eigenvalue_count_by_block[-1]}"
)
