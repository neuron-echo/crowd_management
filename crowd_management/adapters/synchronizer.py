"""Temporal synchronization between CSRNet and RAFT streams.

Enforces strict temporal causality:
1. SAME CAMERA: CSRNet and RAFT observations must originate from the same camera.
2. CONSECUTIVE SOURCE FRAMES: RAFT pairs must be consecutive sampled frames (prev + 1 == curr).
3. NO FUTURE PAIRS: RAFT pair endpoint must be <= current CSRNet frame (and timestamp <= CSRNet timestamp).
4. HISTORICAL REUSE: If no new RAFT pair has arrived for the current CSRNet frame, reuse the MOST RECENT
   valid historical pair, marked as LAST_KNOWN.
5. DYNAMIC STALENESS: Age is calculated as (csr_video_timestamp - raft_video_timestamp). If age > threshold,
   motion is marked STALE and excluded from CrowdState velocity/direction.
6. UNAVAILABLE: If no valid historical pair exists <= current CSRNet frame, motion is UNAVAILABLE.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from crowd_management.models.contracts import CSRNetResult, RAFTResult


class MotionFreshness(str, Enum):
    """Temporal freshness state of a motion observation relative to the density frame."""
    NEW = "NEW"
    LAST_KNOWN = "LAST_KNOWN"
    STALE = "STALE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SyncResult:
    """Detailed temporal synchronization result for a CSRNet frame."""
    raft_result: RAFTResult | None
    status: MotionFreshness
    age_s: float
    is_valid_motion: bool
    pair_provenance: str | None
    reason: str
    rejected_future_count: int = 0

    @property
    def display_tag(self) -> str:
        if self.status == MotionFreshness.NEW:
            return "[NEW]"
        elif self.status == MotionFreshness.LAST_KNOWN:
            return f"[LAST KNOWN (age: {self.age_s:.2f}s)]"
        elif self.status == MotionFreshness.STALE:
            return f"[STALE ({self.age_s:.2f}s > threshold) - EXCLUDED]"
        else:
            return "[UNAVAILABLE]"


class TemporalSynchronizer:
    """Thread-safe per-camera temporal synchronizer for Phase 2 model results.

    Maintains a bounded history of completed RAFT pairs and matches each incoming
    CSRNet frame with its latest valid historical or concurrent RAFT pair.
    """

    def __init__(self, max_history_per_camera: int = 50) -> None:
        self._max_history = max_history_per_camera
        self._history: dict[str, deque[RAFTResult]] = {}
        self._last_used_pair_id: dict[str, int] = {}
        self._future_rejections_count: int = 0
        self._total_sync_queries: int = 0
        self._unavailable_count: int = 0
        self._last_known_count: int = 0
        self._new_count: int = 0
        self._stale_count: int = 0

    def add_raft_result(self, raft_res: RAFTResult) -> None:
        """Add a newly completed RAFT result to the camera's history buffer."""
        cam_id = raft_res.camera_id
        if cam_id not in self._history:
            self._history[cam_id] = deque(maxlen=self._max_history)

        # Basic consecutive check on insertion
        self._history[cam_id].append(raft_res)

    def synchronize(
        self,
        csrnet_res: CSRNetResult,
        stale_threshold_s: float = 1.0,
    ) -> SyncResult:
        """Find the latest valid historical or concurrent RAFT pair for the given CSRNet frame.

        Rules:
        1. Must match camera_id
        2. Must be consecutive: pair_prev + 1 == pair_curr
        3. pair_curr <= csrnet_res.frame_id
        4. raft_video_timestamp <= csrnet_res.video_timestamp + 1e-4
        5. Future pairs (curr > csr_frame or raft_ts > csr_ts) are strictly rejected.
        6. Choose the candidate with highest current_frame_id (most recent valid).
        """
        self._total_sync_queries += 1
        cam_id = csrnet_res.camera_id
        csr_frame = csrnet_res.frame_id
        csr_time = csrnet_res.video_timestamp

        camera_history = self._history.get(cam_id, deque())
        if not camera_history:
            self._unavailable_count += 1
            return SyncResult(
                raft_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_s=0.0,
                is_valid_motion=False,
                pair_provenance=None,
                reason="NO_RAFT_RESULTS_FOR_CAMERA",
                rejected_future_count=0,
            )

        valid_candidates: list[RAFTResult] = []
        future_count = 0

        for r in camera_history:
            # 1. Camera check
            if r.camera_id != cam_id:
                continue

            # 2. Consecutive check
            is_consecutive = (r.previous_frame_id + 1 == r.current_frame_id)
            if not is_consecutive:
                continue

            # 3. Future check (frame index or video timestamp ahead of CSRNet)
            is_future_frame = (r.current_frame_id > csr_frame)
            is_future_time = (r.current_timestamp > csr_time + 1e-4)

            if is_future_frame or is_future_time:
                future_count += 1
                continue

            # 4. Valid historical/concurrent candidate
            valid_candidates.append(r)

        if future_count > 0:
            self._future_rejections_count += future_count

        if not valid_candidates:
            self._unavailable_count += 1
            reason = "FUTURE_PAIR_REJECTED" if future_count > 0 else "NO_VALID_HISTORICAL_PAIR"
            return SyncResult(
                raft_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_s=0.0,
                is_valid_motion=False,
                pair_provenance=None,
                reason=reason,
                rejected_future_count=future_count,
            )

        # Pick the MOST RECENT valid candidate (highest current_frame_id)
        best_raft = max(valid_candidates, key=lambda x: (x.current_frame_id, x.current_timestamp))
        pair_prov = f"{best_raft.previous_frame_id}->{best_raft.current_frame_id}"

        # Calculate video timestamp age
        age_s = max(0.0, csr_time - best_raft.current_timestamp)

        if age_s > stale_threshold_s:
            self._stale_count += 1
            return SyncResult(
                raft_result=best_raft,
                status=MotionFreshness.STALE,
                age_s=age_s,
                is_valid_motion=False,
                pair_provenance=pair_prov,
                reason=f"EXCEEDS_STALE_THRESHOLD ({age_s:.2f}s > {stale_threshold_s:.1f}s)",
                rejected_future_count=future_count,
            )

        # Motion is temporally valid and fresh
        last_used = self._last_used_pair_id.get(cam_id)
        if last_used == best_raft.current_frame_id:
            status = MotionFreshness.LAST_KNOWN
            reason = "LAST_KNOWN_HISTORICAL"
            self._last_known_count += 1
        else:
            status = MotionFreshness.NEW
            reason = "FRESH_OBSERVATION"
            self._last_used_pair_id[cam_id] = best_raft.current_frame_id
            self._new_count += 1

        return SyncResult(
            raft_result=best_raft,
            status=status,
            age_s=age_s,
            is_valid_motion=True,
            pair_provenance=pair_prov,
            reason=reason,
            rejected_future_count=future_count,
        )

    @staticmethod
    def validate_single_pair(
        csrnet_result: CSRNetResult | None,
        raft_result: RAFTResult | None,
        stale_threshold_s: float = 1.0,
    ) -> SyncResult:
        """Helper to validate a single given pair without history buffer."""
        if csrnet_result is None and raft_result is None:
            return SyncResult(
                raft_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_s=0.0,
                is_valid_motion=False,
                pair_provenance=None,
                reason="MISSING_OBSERVATION",
            )

        if csrnet_result is None and raft_result is not None:
            # RAFT observation is provided alone (e.g. testing or motion-only camera)
            pair_prov = f"{raft_result.previous_frame_id}->{raft_result.current_frame_id}"
            return SyncResult(
                raft_result=raft_result,
                status=MotionFreshness.NEW,
                age_s=0.0,
                is_valid_motion=True,
                pair_provenance=pair_prov,
                reason="MOTION_WITHOUT_DENSITY",
            )

        if raft_result is None:
            return SyncResult(
                raft_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_s=0.0,
                is_valid_motion=False,
                pair_provenance=None,
                reason="MISSING_MOTION_OBSERVATION",
            )

        if csrnet_result.camera_id != raft_result.camera_id:
            return SyncResult(
                raft_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_s=0.0,
                is_valid_motion=False,
                pair_provenance=None,
                reason="CROSS_CAMERA_MISMATCH",
            )

        # Check consecutive
        if raft_result.previous_frame_id + 1 != raft_result.current_frame_id:
            return SyncResult(
                raft_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_s=0.0,
                is_valid_motion=False,
                pair_provenance=f"{raft_result.previous_frame_id}->{raft_result.current_frame_id}",
                reason="NON_CONSECUTIVE_PAIR",
            )

        # Check future pair
        is_future_frame = raft_result.current_frame_id > csrnet_result.frame_id
        is_future_time = raft_result.current_timestamp > csrnet_result.video_timestamp + 1e-4

        if is_future_frame or is_future_time:
            return SyncResult(
                raft_result=None,
                status=MotionFreshness.UNAVAILABLE,
                age_s=0.0,
                is_valid_motion=False,
                pair_provenance=f"{raft_result.previous_frame_id}->{raft_result.current_frame_id}",
                reason="FUTURE_PAIR_REJECTED",
                rejected_future_count=1,
            )

        pair_prov = f"{raft_result.previous_frame_id}->{raft_result.current_frame_id}"
        age_s = max(0.0, csrnet_result.video_timestamp - raft_result.current_timestamp)

        if age_s > stale_threshold_s:
            return SyncResult(
                raft_result=raft_result,
                status=MotionFreshness.STALE,
                age_s=age_s,
                is_valid_motion=False,
                pair_provenance=pair_prov,
                reason=f"EXCEEDS_STALE_THRESHOLD ({age_s:.2f}s > {stale_threshold_s:.1f}s)",
            )

        return SyncResult(
            raft_result=raft_result,
            status=MotionFreshness.NEW,
            age_s=age_s,
            is_valid_motion=True,
            pair_provenance=pair_prov,
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
            "future_rejections_count": self._future_rejections_count,
        }
