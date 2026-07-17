"""Names and mappings used by the color-and-shape task.

This module contains the experimental vocabulary. It does not simulate neural
activity. Keeping these names separate makes it easy to see which parts describe
the task and which parts describe the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Feature(IntEnum):
    """Meaning of each position in the six-number feature vector."""

    GREEN = 0
    BLUE = 1
    SQUARE = 2
    CIRCLE = 3
    LEFT = 4
    RIGHT = 5


class Task(str, Enum):
    """The stimulus dimension that determines the correct response."""

    COLOR = "color"
    SHAPE = "shape"


COLOR_FEATURES = (Feature.GREEN, Feature.BLUE)
SHAPE_FEATURES = (Feature.SQUARE, Feature.CIRCLE)
RESPONSE_FEATURES = (Feature.LEFT, Feature.RIGHT)

# The neural model uses the same three pairs for within-pair competition.
COMPETING_FEATURE_GROUPS = (
    COLOR_FEATURES,
    SHAPE_FEATURES,
    RESPONSE_FEATURES,
)

FEATURES_BY_TASK = {
    Task.COLOR: COLOR_FEATURES,
    Task.SHAPE: SHAPE_FEATURES,
}

IRRELEVANT_FEATURES_BY_TASK = {
    Task.COLOR: SHAPE_FEATURES,
    Task.SHAPE: COLOR_FEATURES,
}

# Green and square map to left. Blue and circle map to right.
RESPONSE_BY_FEATURE = {
    Feature.GREEN: Feature.LEFT,
    Feature.BLUE: Feature.RIGHT,
    Feature.SQUARE: Feature.LEFT,
    Feature.CIRCLE: Feature.RIGHT,
}


# A dataclass generates the routine code needed to store these named values.
@dataclass(frozen=True)
class Stimulus:
    """One color-shape combination presented on a trial."""

    color: Feature
    shape: Feature

    def __post_init__(self) -> None:
        if self.color not in COLOR_FEATURES:
            raise ValueError(f"{self.color!r} is not a color feature")
        if self.shape not in SHAPE_FEATURES:
            raise ValueError(f"{self.shape!r} is not a shape feature")

    def relevant_feature(self, task: Task) -> Feature:
        """Select the color or shape according to the current task."""

        if task == Task.COLOR:
            return self.color
        if task == Task.SHAPE:
            return self.shape
        raise ValueError(f"Unknown task: {task!r}")


# Every possible combination occurs in every block of the experiment.
ALL_STIMULI = (
    Stimulus(Feature.GREEN, Feature.SQUARE),
    Stimulus(Feature.GREEN, Feature.CIRCLE),
    Stimulus(Feature.BLUE, Feature.SQUARE),
    Stimulus(Feature.BLUE, Feature.CIRCLE),
)


def correct_response(task: Task, stimulus: Stimulus) -> Feature:
    """Return the response selected by the current rule."""

    relevant_feature = stimulus.relevant_feature(task)
    return RESPONSE_BY_FEATURE[relevant_feature]


def is_congruent(stimulus: Stimulus) -> bool:
    """Return whether color and shape indicate the same response."""

    color_response = correct_response(Task.COLOR, stimulus)
    shape_response = correct_response(Task.SHAPE, stimulus)
    return color_response == shape_response
