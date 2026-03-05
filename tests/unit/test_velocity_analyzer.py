
from core.behavior.base_analyzer import BehaviorLabel


def test_behavior_labels():

    labels = [l.value for l in BehaviorLabel]

    assert "normal" in labels
    assert "panic" in labels
    assert "running" in labels
