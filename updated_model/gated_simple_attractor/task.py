"""Names and mappings used by the no-cue colour-and-shape switching task.

This module holds the experimental vocabulary. It does not simulate neural
activity. Keeping the names here separate from model.py makes it easy to see
which parts describe the task and which parts describe the network.

Unlike cued_attractor/gated_attractor, there is no cue: the vocabulary is a
fixed six-unit layout (no per-rule cue block), since the rule is signalled by
which single feature the experiment drives, not by an extra input unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Feature(IntEnum):
    """Meaning of the four feature-vector positions that never move."""

    GREEN = 0
    BLUE = 1
    SQUARE = 2
    CIRCLE = 3


class Task(str, Enum):
    """The stimulus dimension that determines the correct response (the rule)."""

    COLOR = 'color'
    SHAPE = 'shape'


COLOR_FEATURES = (Feature.GREEN, Feature.BLUE)
SHAPE_FEATURES = (Feature.SQUARE, Feature.CIRCLE)

FEATURES_BY_TASK = {
    Task.COLOR: COLOR_FEATURES,
    Task.SHAPE: SHAPE_FEATURES,
}


# A dataclass generates the routine code needed to store these named values.
@dataclass(frozen=True)
class Vocabulary:
    """Feature-vector layout: colour/shape at fixed indices 0-3, then the two
    action indices. No cue block, so nothing shifts and nothing needs to be
    parameterized.
    """

    number_of_features: int
    response_features: tuple[int, int]
    response_by_feature: dict[Feature, int]
    competing_feature_groups: tuple[tuple[int, ...], ...]


def build_vocabulary() -> Vocabulary:
    """Lay out the six fixed feature positions: no cues, nothing shifts.

    Order: [green, blue, square, circle, action1, action2].
    """

    action_1, action_2 = 4, 5
    response_features = (action_1, action_2)

    return Vocabulary(
        number_of_features=action_2 + 1,
        response_features=response_features,
        response_by_feature={
            Feature.GREEN: action_1, Feature.BLUE: action_2,
            Feature.SQUARE: action_1, Feature.CIRCLE: action_2,
        },
        # The neural model reuses these groups for within-group lateral
        # competition. Colour, shape and action are pairs; no cue group.
        competing_feature_groups=(COLOR_FEATURES, SHAPE_FEATURES, response_features),
    )


# A dataclass generates the routine code needed to store these named values.
@dataclass(frozen=True)
class Stimulus:
    """One colour-shape combination presented on a trial."""

    color: Feature
    shape: Feature

    def __post_init__(self) -> None:
        if self.color not in COLOR_FEATURES:
            raise ValueError(f'{self.color!r} is not a colour feature')
        if self.shape not in SHAPE_FEATURES:
            raise ValueError(f'{self.shape!r} is not a shape feature')

    def relevant_feature(self, task: Task) -> Feature:
        """Select the colour or shape according to the current rule.

        The only one of the pair the experiment ever drives -- see
        experiment.py's _relevant_feature_vector.
        """

        if task == Task.COLOR:
            return self.color
        if task == Task.SHAPE:
            return self.shape
        raise ValueError(f'Unknown task: {task!r}')

    def irrelevant_feature(self, task: Task) -> Feature:
        """Select the colour or shape the current rule ignores.

        Never driven by the experiment; used only by analysis.py to check
        that it actually settles near baseline.
        """

        if task == Task.COLOR:
            return self.shape
        if task == Task.SHAPE:
            return self.color
        raise ValueError(f'Unknown task: {task!r}')


# Every combination occurs in every block of the experiment. Row order matches
# the old stimuli array (green square, green circle, blue square, blue circle).
ALL_STIMULI = (
    Stimulus(Feature.GREEN, Feature.SQUARE),
    Stimulus(Feature.GREEN, Feature.CIRCLE),
    Stimulus(Feature.BLUE, Feature.SQUARE),
    Stimulus(Feature.BLUE, Feature.CIRCLE),
)


def correct_response(vocabulary: Vocabulary, task: Task, stimulus: Stimulus) -> int:
    """Return the action feature index selected by the current rule."""

    relevant_feature = stimulus.relevant_feature(task)
    return vocabulary.response_by_feature[relevant_feature]


def is_congruent(stimulus: Stimulus) -> bool:
    """Return whether colour and shape indicate the same response.

    Congruent stimuli (green square, blue circle) get the same action from both
    rules; incongruent stimuli (green circle, blue square) get opposite actions.
    Kept for comparability with the sibling packages, but since the irrelevant
    feature is never shown here, congruency has no route to affect a trial's
    dynamics -- switch_contrast_by_kind/incongruence_contrast_by_kind should
    come out null, which is the expected negative control, not a bug.
    """

    return (stimulus.color == Feature.GREEN) == (stimulus.shape == Feature.SQUARE)
