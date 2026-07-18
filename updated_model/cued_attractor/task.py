"""Names and mappings used by the cued colour-and-shape switching task.

This module holds the experimental vocabulary. It does not simulate neural
activity. Keeping the names here separate from model.py makes it easy to see
which parts describe the task and which parts describe the network.

It extends the published six-feature vocabulary with four cue units, so the
rule can be signalled by a cue on the input rather than re-taught every block.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class Feature(IntEnum):
    """Meaning of the four feature-vector positions that never move.

    Cue and action positions shift with num_cues_per_rule, so they aren't fixed
    enum members; see build_vocabulary below.
    """

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

IRRELEVANT_FEATURES_BY_TASK = {
    Task.COLOR: SHAPE_FEATURES,
    Task.SHAPE: COLOR_FEATURES,
}


# A dataclass generates the routine code needed to store these named values.
@dataclass(frozen=True)
class Vocabulary:
    """Feature-vector layout for a given number of cues per rule.

    Colour/shape stay fixed at indices 0-3 (Feature enum); the cue block (2
    rules x num_cues_per_rule) and the two action indices after it shift
    position with num_cues_per_rule, so they're plain ints here rather than
    enum members.
    """

    num_cues_per_rule: int
    number_of_features: int
    cues_by_task: dict[Task, tuple[int, ...]]
    response_features: tuple[int, int]
    response_by_feature: dict[Feature, int]
    competing_feature_groups: tuple[tuple[int, ...], ...]


def build_vocabulary(num_cues_per_rule: int = 2) -> Vocabulary:
    """Lay out cue and action positions after the four fixed colour/shape units.

    Order matches the flat attractor_rnn.py script at num_cues_per_rule=2:
    [green, blue, square, circle, cueA1, cueA2, cueB1, cueB2, action1, action2].
    """

    if num_cues_per_rule <= 0:
        raise ValueError('num_cues_per_rule must be positive')

    cue_start = 4
    color_cues = tuple(range(cue_start, cue_start + num_cues_per_rule))
    shape_cues = tuple(
        range(cue_start + num_cues_per_rule, cue_start + 2 * num_cues_per_rule)
    )
    action_1 = cue_start + 2 * num_cues_per_rule
    action_2 = action_1 + 1
    response_features = (action_1, action_2)

    return Vocabulary(
        num_cues_per_rule=num_cues_per_rule,
        number_of_features=action_2 + 1,
        cues_by_task={Task.COLOR: color_cues, Task.SHAPE: shape_cues},
        response_features=response_features,
        response_by_feature={
            Feature.GREEN: action_1, Feature.BLUE: action_2,
            Feature.SQUARE: action_1, Feature.CIRCLE: action_2,
        },
        # The neural model reuses these groups for within-group lateral
        # competition. Colour, shape and action are pairs; all cues form a
        # single group, so cue competition is winner-take-all across all of
        # them (only one cue is ever shown per trial).
        competing_feature_groups=(
            COLOR_FEATURES, SHAPE_FEATURES, color_cues + shape_cues, response_features,
        ),
    )


def cues_for_task(vocabulary: Vocabulary, task: Task) -> tuple[int, ...]:
    """Return the cues that signal the given rule."""

    return vocabulary.cues_by_task[task]


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
        """Select the colour or shape according to the current rule."""

        if task == Task.COLOR:
            return self.color
        if task == Task.SHAPE:
            return self.shape
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
    This pairing is fixed regardless of num_cues_per_rule, so it's checked
    directly rather than via correct_response (which needs a Vocabulary).
    """

    return (stimulus.color == Feature.GREEN) == (stimulus.shape == Feature.SQUARE)
