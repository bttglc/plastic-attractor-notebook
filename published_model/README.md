# Published Blocked Model

This folder contains the cleaned multi-file implementation of the model used in the paper. It is our working control model for later extensions.

## Run it

From the repository's top-level folder:

```bash
cd published_model
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
python3 examples/run_baseline.py
```

The main workflow uses two functions:

```python
from plastic_attractor import run_baseline, summarize_behavior

result = run_baseline(seed=0, number_of_blocks=20)
summary = summarize_behavior(result.trials)
```

## File map

- `plastic_attractor/task.py`: stimuli, tasks, and response mappings
- `plastic_attractor/model.py`: neural activity and Hebbian learning
- `plastic_attractor/experiment.py`: instructions, trials, blocks, and perturbations
- `plastic_attractor/analysis.py`: accuracy and reaction-time summaries
- `examples/run_baseline.py`: the shortest complete example
- `tests/`: checks that the model retains its expected behavior

Start with the example, then read [`task.py`](plastic_attractor/task.py), [`model.py`](plastic_attractor/model.py), [`experiment.py`](plastic_attractor/experiment.py), and [`analysis.py`](plastic_attractor/analysis.py) in that order.

This version implements the published blocked task. Trial-wise cue-driven switching will be added later.
