"""Temporal synchronization between CSRNet and ByteTrack streams.

ByteTrack operates on individual frames (not pairs), so synchronization is
simpler than the former RAFT-based approach:

Rules:
1. SAME CAMERA: CSRNet and ByteTrack observations must share camera_id.
2. NO FUTURE FRAMES: ByteTrack frame_id must be <= current CSRNet frame_id.
3. HISTORICAL REUSE: If no ByteTrack result exists for the exact CSRNet frame,
   reuse the most recent valid historical result, marked as LAST_KNOWN.
4. STALENESS: Age is computed as (csr_frame_id - bytetrack_frame_id). If the
   frame gap exceeds the staleness threshold, motion is marked STALE.
5. UNAVAILABLE: If no valid historical result exists, motion is UNAVAILABLE.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from crowd_management.models.contracts import ByteTrackResult, CSRNetResult


class MotionFreshness(str, Enum):
    """Temporal freshness state of a motion observation relative to the density frame."""
    NEW = "NEW"
    LAST_KNOWN = "LAST_KNOWN"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SyncResult:
    """Detailed temporal synchronization result for a CSRNet frame."""
    bytetrack_result: ByteTrackResult | None
    status: MotionFreshness
    age_frames: int           # frame_id delta between CSRNet and ByteTrack
    is_valid_motion: bool
    pair_provenance: str | None  # e.g. "bytetrack@frame_42"
    reason: str

    @property
    def display_tag(self) -> str:
        if self.status == MotionFreshness.NEW:
            return "[NEW]"
        elif self.status == MotionFreshness.LAST_KNOWN:
            return f"[LAST KNOWN (gap: {self.age_frames} frames)]"
        elif self.status == MotionFreshness.STALE:
            return f"[STALE (gap {self.age_frames} frames > threshold) - EXCLUDED]"
        else:
            return "[UNAVAILABLE]"


class TemporalSynchronizer:
    """Thread-safe per-camera temporal synchronizer for ByteTrack + CSRNet results.

    Maintains a bounded FIFO history of ByteTrack results per camera and, for
    each incoming CSRNet frame, finds the most recent ByteTrack result whose
    frame_id does not exceed the CSRNet frame_id.
    """

    def __init__(
        self,
        max_history_per_camera: int = 50,
        stale_frame_gap: int = 10,
    ) -> None:
        """
        Args:
            max_history_per_camera: Maximum ByteTrackResult history per camera.
            stale_frame_gap: Frame-id delta above which motion is considered STALE.
        """
        self._max_history = max_history_per_camera
        self._stale_frame_gap = stale_frame_gap
        self._history: dict[str, deque[ByteTrackResult]] = {}
        self._last_used_frame_id: dict[str, int] = {}

        # Telemetry counters
        self._total_sync_queries: int = 0
        self._new_count: int = 0
        self._last_known_count: int = 0
        self._stale_count: int = 0
        self._unavailable_count: int = 0

    def add_bytetrack_result(self, bt_res: ByteTrackResult) -> None:
        """Add a newly completed ByteTrack result to the camera's history buffer."""
        cam_id = bt_res.camera_id
        if cam_id not in self._history:
            self._history[cam_id] = deque(maxlen=self._max_history)
        self._history[cam_id].append(bt_res)

    def synchronize(
        self,
        csrnet_res: CSRNetResult,
        stale_frame_gap: int | None = None,
    ) -> SyncResult:
        """Find the latest valid ByteTrack result for the given CSRNet frame.

        Args:
            csrnet_res: CSRNet density estimation result to match against.
            stale_frame_gap: Override stale threshold (frames). Uses constructor
                             default if None.

        Returns:
            SyncResult describing the matched ByteTrack result and its freshness.
        """
        self._total_sync_queries += 1
        threshold = stale_frame_gap if stale_frame_gap is not None else self._stale_frame_gap
        cam_id = csrnet_res.camera_id
        csr_frame = csrnet_res.frame_id

        camera_history = self._history.get(cam_id, deque())
        if not camera_history:
            self._unavailable_count += 1
            return SyncResult(
                bytetrack_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_frames=0,
                is_valid_motion=False,
                pair_provenance=None,
                reason="NO_BYTETRACK_RESULTS_FOR_CAMERA",
            )

        # Find the most recent ByteTrack result with frame_id <= csr_frame
        valid_candidates = [r for r in camera_history
                            if r.camera_id == cam_id and r.frame_id <= csr_frame]

        if not valid_candidates:
            self._unavailable_count += 1
            return SyncResult(
                bytetrack_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_frames=0,
                is_valid_motion=False,
                pair_provenance=None,
                reason="NO_VALID_HISTORICAL_BYTETRACK",
            )

        # Pick the most recent (highest frame_id)
        best = max(valid_candidates, key=lambda r: r.frame_id)
        age_frames = csr_frame - best.frame_id
        prov = f"bytetrack@frame_{best.frame_id}"

        # Staleness check (frame gap)
        if age_frames > threshold:
            self._stale_count += 1
            return SyncResult(
                bytetrack_result=best,
                status=MotionFreshness.STALE,
                age_frames=age_frames,
                is_valid_motion=False,
                pair_provenance=prov,
                reason=f"EXCEEDS_STALE_THRESHOLD (gap {age_frames} > {threshold})",
            )

        # Determine NEW vs LAST_KNOWN
        last_used = self._last_used_frame_id.get(cam_id, -1)
        if best.frame_id != last_used:
            status = MotionFreshness.NEW
            reason = "FRESH_OBSERVATION"
            self._last_used_frame_id[cam_id] = best.frame_id
            self._new_count += 1
        else:
            status = MotionFreshness.LAST_KNOWN
            reason = "LAST_KNOWN_HISTORICAL"
            self._last_known_count += 1

        return SyncResult(
            bytetrack_result=best,
            status=status,
            age_frames=age_frames,
            is_valid_motion=True,
            pair_provenance=prov,
            reason=reason,
        )

    @staticmethod
    def validate_single_pair(
        csrnet_result: CSRNetResult | None,
        bytetrack_result: ByteTrackResult | None,
        stale_frame_gap: int = 10,
    ) -> SyncResult:
        """Validate a single given pair without a history buffer (for testing/demo)."""
        if csrnet_result is None and bytetrack_result is None:
            return SyncResult(
                bytetrack_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_frames=0,
                is_valid_motion=False,
                pair_provenance=None,
                reason="MISSING_OBSERVATION",
            )

        if bytetrack_result is None:
            return SyncResult(
                bytetrack_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_frames=0,
                is_valid_motion=False,
                pair_provenance=None,
                reason="MISSING_MOTION_OBSERVATION",
            )

        if csrnet_result is None:
            prov = f"bytetrack@frame_{bytetrack_result.frame_id}"
            return SyncResult(
                bytetrack_result=bytetrack_result,
                status=MotionFreshness.NEW,
                age_frames=0,
                is_valid_motion=True,
                pair_provenance=prov,
                reason="MOTION_WITHOUT_DENSITY",
            )

        if csrnet_result.camera_id != bytetrack_result.camera_id:
            return SyncResult(
                bytetrack_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_frames=0,
                is_valid_motion=False,
                pair_provenance=None,
                reason="CROSS_CAMERA_MISMATCH",
            )

        if bytetrack_result.frame_id > csrnet_result.frame_id:
            return SyncResult(
                bytetrack_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_frames=0,
                is_valid_motion=False,
                pair_provenance=f"bytetrack@frame_{bytetrack_result.frame_id}",
                reason="FUTURE_BYTETRACK_REJECTED",
            )

        age_frames = csrnet_result.frame_id - bytetrack_result.frame_id
        prov = f"bytetrack@frame_{bytetrack_result.frame_id}"

        if age_frames > stale_frame_gap:
            return SyncResult(
                bytetrack_result=bytetrack_result,
                status=MotionFreshness.STALE,
                age_frames=age_frames,
                is_valid_motion=False,
                pair_provenance=prov,
                reason=f"EXCEEDS_STALE_THRESHOLD (gap {age_frames} > {stale_frame_gap})",
            )

        return SyncResult(
            bytetrack_result=bytetrack_result,
            status=MotionFreshness.NEW,
            age_frames=age_frames,
            is_valid_motion=True,
            pair_provenance=prov,
            reason="FRESH_OBSERVATION",
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_sync_queries": self._total_sync_queries,
            "new_count": self._new_count,
            "last_known_count": self._last_known_count,
            "stale_count": self._stale_count,
            "unavailable_count": self._unavailable_count,
        }
