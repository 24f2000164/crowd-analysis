"""
core/tracking/track_state.py
=============================
Track lifecycle state machine.

A track moves through well-defined states:

    Tentative  ──(min_hits reached)──►  Confirmed
       │                                    │
       └──(missed too long)──►  Lost ──────►┘
                                    │
                                    └──(max_age exceeded)──► Deleted

Only Confirmed tracks are surfaced to downstream pipeline stages.
Tentative and Lost tracks are held internally for re-association.
"""

from __future__ import annotations

from enum import IntEnum, unique


@unique
class TrackState(IntEnum):
    """
    Lifecycle states for a single tracked person.

    IntEnum is used deliberately so states can be compared with ``<`` / ``>``
    in simple guard conditions.
    """

    Tentative = 1
    """
    Newly created track.  Not yet reliable enough to surface to downstream
    consumers.  Transitions to Confirmed once ``min_hits`` consecutive
    detections are received.
    """

    Confirmed = 2
    """
    Track has been matched consistently and is considered reliable.
    Surfaced to the behavior analysis pipeline.
    """

    Lost = 3
    """
    Track was not matched in the current frame.  Held for re-association
    for up to ``max_age`` frames before deletion.
    """

    Deleted = 4
    """
    Terminal state.  Track has exceeded ``max_age`` without a match.
    Will be purged from the active track registry on the next update cycle.
    """

    @property
    def is_active(self) -> bool:
        """True for states that are still candidates for detection matching."""
        return self in (TrackState.Tentative, TrackState.Confirmed, TrackState.Lost)

    @property
    def is_output_ready(self) -> bool:
        """True only for Confirmed — the only state surfaced downstream."""
        return self == TrackState.Confirmed
