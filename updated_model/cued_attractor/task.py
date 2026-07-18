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
    """Meaning of each position in the ten-number feature vector.

    Order matches the flat attractor_rnn.py script:
    [green, blue, square, circle, cueA1, cueA2, cueB1, cueB2, action1, action2].
    The two cue pairs are dissociable from the rule they signal (see RULE_BY_CUE).
    """

    GREEN = 0
    BLUE = 1
    SQUARE = 2
    CIRCLE = 3
    CUE_A1 = 4
    CUE_A2 = 5
    CUE_B1 = 6
    CUE_B2 = 7
    ACTION_1 = 8
    ACTION_2 = 9


class Task(str, Enum):
    """The stimulus dimension that determines the correct response (the rule)."""

    COLOR = 'color'
    SHAPE = 'shape'


class Cue(IntEnum):
    """The four cue units, two per rule.

    Values equal the Feature index of the matching cue unit, so a Cue can index
    straight into the feature vector.
    """

    A1 = Feature.CUE_A1
    A2 = Feature.CUE_A2
    B1 = Feature.CUE_B1
    B2 = Feature.CUE_B2


COLOR_FEATURES = (Feature.GREEN, Feature.BLUE)
SHAPE_FEATURES = (Feature.SQUARE, Feature.CIRCLE)
CUE_FEATURES = (Feature.CUE_A1, Feature.CUE_A2, Feature.CUE_B1, Feature.CUE_B2)
RESPONSE_FEATURES = (Feature.ACTION_1, Feature.ACTION_2)

# The neural model reuses these groups for within-group lateral competition. The
# colour, shape and action groups are pairs; the four cues form a single group,
# so cue competition is winner-take-all across all four (only one cue is ever
# shown per trial).
COMPETING_FEATURE_GROUPS = (
    COLOR_FEATURES,
    SHAPE_FEATURES,
    CUE_FEATURES,
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

# Cue -> rule it signals. A-cues mean 'attend colour', B-cues mean 'attend
# shape'. Two cues per rule keep cue identity and abstract rule dissociable
# (old cue_rule = [0, 0, 1, 1]).
RULE_BY_CUE = {
    Cue.A1: Task.COLOR,
    Cue.A2: Task.COLOR,
    Cue.B1: Task.SHAPE,
    Cue.B2: Task.SHAPE,
}

CUES_BY_TASK = {
    Task.COLOR: (Cue.A1, Cue.A2),
    Task.SHAPE: (Cue.B1, Cue.B2),
}

# Green and square map to action 1. Blue and circle map to action 2.
RESPONSE_BY_FEATURE = {
    Feature.GREEN: Feature.ACTION_1,
    Feature.BLUE: Feature.ACTION_2,
    Feature.SQUARE: Feature.ACTION_1,
    Feature.CIRCLE: Feature.ACTION_2,
}


def cues_for_task(task: Task) -> tuple[Cue, ...]:
    """Return the two cues that signal the given rule."""

    return CUES_BY_TASK[task]


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


def correct_response(task: Task, stimulus: Stimulus) -> Feature:
    """Return the action feature selected by the current rule."""

    relevant_feature = stimulus.relevant_feature(task)
    return RESPONSE_BY_FEATURE[relevant_feature]


def is_congruent(stimulus: Stimulus) -> bool:
    """Return whether colour and shape indicate the same response.

    Congruent stimuli (green square, blue circle) get the same action from both
    rules; incongruent stimuli (green circle, blue square) get opposite actions.
    """

    color_response = correct_response(Task.COLOR, stimulus)
    shape_response = correct_response(Task.SHAPE, stimulus)
    return color_response == shape_response
