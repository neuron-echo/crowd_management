"""Phase 2 → Phase 3 adapter boundary.

THIS IS THE ONLY FILE IN THE PROJECT that imports both Phase 2 model-specific
contracts (CSRNetResult, ByteTrackResult) and Phase 3 model-agnostic contracts
(CrowdState).

If CSRNet or ByteTrack are replaced, only this file needs to change.
The Decision Engine, Venue Configuration, and all Phase 3 logic remain untouched.

Dependency direction:
    Phase 2 (CSRNetResult, ByteTrackResult)
        ↓
    Phase2Adapter (this file)
        ↓
    CrowdState (model-agnostic)
        ↓
    Decision Engine (no model imports)
"""

from __future__ import annotations

import time

from crowd_management.adapters.synchronizer import MotionFreshness, SyncResult, TemporalSynchronizer
from crowd_management.decision.config import DecisionConfig
from crowd_management.decision.contracts import CrowdState, Trend
from crowd_management.decision.fusion import build_crowd_state

# These are the ONLY Phase 2 imports in all of Phase 3.
from crowd_management.models.contracts import ByteTrackResult, CSRNetResult


class Phase2Adapter:
    """Adapter that fuses CSRNetResult + ByteTrackResult into a model-agnostic CrowdState.

    Handles:
    - Missing CSRNet data (density unavailable)
    - Missing ByteTrack data (motion unavailable)
    - Stale observations (frame gap tracking)
    - Future ByteTrack frame rejection (temporal causality)
    - Historical ByteTrack reuse (LAST_KNOWN)
    - Camera→gate resolution
    - Never mixes observations from different cameras

    Usage:
        adapter = Phase2Adapter(config=DecisionConfig())
        crowd_state = adapter.fuse(
            csrnet_result=csr_res,            # or None
            bytetrack_result=bytetrack_res,   # or None
            gate_id="G1",
            gate_capacity=500,
        )
    """

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self._config = config or DecisionConfig()

    def fuse(
        self,
        csrnet_result: CSRNetResult | None,
        bytetrack_result: ByteTrackResult | None,
        gate_id: str,
        gate_capacity: int,
        trend: Trend = Trend.STABLE,
        stale_frame_gap: int = 10,
        sync_result: SyncResult | None = None,
    ) -> CrowdState:
        """Fuse Phase 2 model results into a model-agnostic CrowdState.

        Args:
            csrnet_result:   CSRNet density estimation result, or None if unavailable.
            bytetrack_result: ByteTrack motion result, or None if unavailable.
            gate_id:         Gate identifier from venue configuration.
            gate_capacity:   Maximum capacity of this gate.
            trend:           Pre-computed trend from TrendTracker.
            stale_frame_gap: Maximum frame-id gap for motion to be considered fresh.
            sync_result:     Optional pre-computed temporal synchronization result.

        Returns:
            Model-agnostic CrowdState with valid/fresh motion or explicitly marked
            stale/unavailable motion.

        Raises:
            ValueError: If results are from different cameras.
        """
        now = time.time()

        # Cross-camera safety check
        if csrnet_result is not None and bytetrack_result is not None:
            if csrnet_result.camera_id != bytetrack_result.camera_id:
                raise ValueError(
                    f"Cannot fuse results from different cameras: "
                    f"CSRNet={csrnet_result.camera_id}, ByteTrack={bytetrack_result.camera_id}"
                )

        # Extract density observation
        has_density = csrnet_result is not None
        crowd_count = 0.0
        source_frame_id = None
        result_age_s = 0.0
        timestamp = now

        if csrnet_result is not None:
            crowd_count = csrnet_result.crowd_count
            source_frame_id = csrnet_result.frame_id
            result_age_s = max(0.0, now - csrnet_result.result_timestamp)
            timestamp = csrnet_result.result_timestamp

        # Synchronize motion observation
        if sync_result is None:
            sync_result = TemporalSynchronizer.validate_single_pair(
                csrnet_result=csrnet_result,
                bytetrack_result=bytetrack_result,
                stale_frame_gap=stale_frame_gap,
            )

        has_motion = sync_result.is_valid_motion
        bt = sync_result.bytetrack_result

        velocity = bt.median_velocity if (has_motion and bt) else 0.0
        direction = bt.dominant_direction_deg if (has_motion and bt) else 0.0
        flow_consistency = bt.flow_consistency if (has_motion and bt) else 0.0
        motion_status = sync_result.status.value
        bytetrack_provenance = sync_result.pair_provenance
        bytetrack_age_frames = sync_result.age_frames
        is_motion_fresh = has_motion

        # Calculate confidence
        if has_motion and bt:
            confidence = bt.motion_confidence
            if csrnet_result is not None:
                # Blend: motion confidence weighted with density confidence (1.0)
                confidence = min(1.0, (confidence + 1.0) / 2.0)
        elif sync_result.status == MotionFreshness.STALE and bt:
            confidence = bt.motion_confidence * 0.7
        else:
            confidence = 0.8 if has_density else 0.5

        return build_crowd_state(
            gate_id=gate_id,
            gate_capacity=gate_capacity,
            crowd_count=crowd_count,
            velocity=velocity,
            direction=direction,
            flow_consistency=flow_consistency,
            confidence=confidence,
            source_frame_id=source_frame_id,
            result_age_s=result_age_s,
            has_density=has_density,
            has_motion=has_motion,
            motion_status=motion_status,
            raft_pair_provenance=bytetrack_provenance,    # field reused for provenance
            raft_result_age_s=float(bytetrack_age_frames),
            is_motion_fresh=is_motion_fresh,
            trend=trend,
            config=self._config,
            timestamp=timestamp,
        )
