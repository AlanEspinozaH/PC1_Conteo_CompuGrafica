from counter import LineCounter
from tracker import CentroidTracker


def detection(cx, cy):
    return {"cx": cx, "cy": cy}


def test_missing_tracks_are_retained_but_not_reported_as_visible():
    tracker = CentroidTracker(max_distance=20, max_disappeared=2)

    first = tracker.update([detection(10, 10)])
    missing = tracker.update([])
    recovered = tracker.update([detection(12, 11)])

    assert first == {1: (10, 10)}
    assert missing == {}
    assert recovered == {1: (12, 11)}


def test_only_one_track_is_assigned_to_each_detection():
    tracker = CentroidTracker(max_distance=50, max_disappeared=2)

    tracker.update([detection(10, 10), detection(40, 10)])
    visible = tracker.update([detection(12, 10), detection(38, 10)])

    assert visible == {1: (12, 10), 2: (38, 10)}


def test_counter_does_not_advance_without_visible_observation():
    tracker = CentroidTracker(max_distance=50, max_disappeared=2)
    counter = LineCounter(line_y=50, margin=5, min_seen_frames=2)

    counter.update(tracker.update([detection(10, 30)]))
    counter.update(tracker.update([]))
    events = counter.update(tracker.update([detection(10, 70)]))

    assert len(events) == 1
    assert events[0]["id"] == 1
    assert counter.seen_frames[1] == 2
