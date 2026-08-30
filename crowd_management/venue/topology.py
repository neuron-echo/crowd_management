"""Venue topology graph for gate connectivity, distance, and adjacency queries.

This module provides an in-memory graph representation of the venue that the
Decision Engine uses for routing. It has NO dependencies on torch, cv2, or
any vision model code.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from crowd_management.venue.models import VenueConfig


@dataclass
class GateNode:
    """In-memory representation of a gate in the topology graph."""

    gate_id: str
    capacity: int
    position_index: int
    camera_id: str | None
    name: str | None


@dataclass
class Edge:
    """Directed edge between two gates."""

    from_gate: str
    to_gate: str
    distance_m: float


class VenueTopology:
    """Graph-based venue topology for gate connectivity and routing queries.

    Built from a VenueConfig. Provides:
    - Gate lookup by ID
    - Camera → gate resolution
    - Adjacency enumeration
    - Shortest distance between connected gates
    - Connected gate enumeration for routing candidates
    """

    def __init__(self, config: VenueConfig) -> None:
        self._config = config
        self._gates: dict[str, GateNode] = {}
        self._adjacency: dict[str, list[Edge]] = defaultdict(list)
        self._camera_to_gate: dict[str, str] = {}
        self._gate_to_camera: dict[str, str | None] = {}
        self._all_distances: dict[tuple[str, str], float] = {}

        self._build(config)

    def _build(self, config: VenueConfig) -> None:
        """Build the topology graph from venue configuration."""
        for gate_cfg in config.gates:
            self._gates[gate_cfg.gate_id] = GateNode(
                gate_id=gate_cfg.gate_id,
                capacity=gate_cfg.capacity,
                position_index=gate_cfg.position_index,
                camera_id=gate_cfg.camera_id,
                name=gate_cfg.name,
            )
            if gate_cfg.camera_id is not None:
                self._camera_to_gate[gate_cfg.camera_id] = gate_cfg.gate_id
                self._gate_to_camera[gate_cfg.gate_id] = gate_cfg.camera_id
            else:
                self._gate_to_camera[gate_cfg.gate_id] = None

        # Build adjacency from effective connections (explicit or auto-derived)
        connections = config.get_effective_connections()
        for conn in connections:
            edge_fwd = Edge(conn.from_gate, conn.to_gate, conn.distance_m)
            edge_bwd = Edge(conn.to_gate, conn.from_gate, conn.distance_m)
            if not any(e.to_gate == conn.to_gate for e in self._adjacency[conn.from_gate]):
                self._adjacency[conn.from_gate].append(edge_fwd)
            if not any(e.to_gate == conn.from_gate for e in self._adjacency[conn.to_gate]):
                self._adjacency[conn.to_gate].append(edge_bwd)
            self._all_distances[(conn.from_gate, conn.to_gate)] = conn.distance_m
            self._all_distances[(conn.to_gate, conn.from_gate)] = conn.distance_m

        # Compute shortest paths between all pairs (Floyd-Warshall on small graph)
        self._compute_all_pairs_shortest()

    def _compute_all_pairs_shortest(self) -> None:
        """Compute shortest path distances between all gate pairs using Floyd-Warshall."""
        gate_ids = list(self._gates.keys())
        INF = float("inf")

        dist: dict[tuple[str, str], float] = {}
        for g1 in gate_ids:
            for g2 in gate_ids:
                if g1 == g2:
                    dist[(g1, g2)] = 0.0
                elif (g1, g2) in self._all_distances:
                    dist[(g1, g2)] = self._all_distances[(g1, g2)]
                else:
                    dist[(g1, g2)] = INF

        for k in gate_ids:
            for i in gate_ids:
                for j in gate_ids:
                    if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                        dist[(i, j)] = dist[(i, k)] + dist[(k, j)]

        self._shortest_distances = dist

    @property
    def gate_ids(self) -> list[str]:
        """Return list of all gate IDs in position_index order."""
        sorted_gates = sorted(self._gates.values(), key=lambda g: g.position_index)
        return [g.gate_id for g in sorted_gates]

    @property
    def gate_count(self) -> int:
        return len(self._gates)

    @property
    def venue_id(self) -> str:
        return self._config.id

    @property
    def venue_name(self) -> str:
        return self._config.name

    def get_gate(self, gate_id: str) -> GateNode:
        """Lookup a gate by ID. Raises KeyError if not found."""
        if gate_id not in self._gates:
            raise KeyError(f"Unknown gate_id: {gate_id}")
        return self._gates[gate_id]

    def get_gate_capacity(self, gate_id: str) -> int:
        return self.get_gate(gate_id).capacity

    def gate_for_camera(self, camera_id: str) -> str | None:
        """Resolve camera_id to gate_id. Returns None if unmapped."""
        return self._camera_to_gate.get(camera_id)

    def camera_for_gate(self, gate_id: str) -> str | None:
        """Resolve gate_id to camera_id. Returns None if unmonitored."""
        return self._gate_to_camera.get(gate_id)

    def neighbors(self, gate_id: str) -> list[str]:
        """Return directly adjacent gate IDs."""
        seen: set[str] = set()
        res: list[str] = []
        for e in self._adjacency.get(gate_id, []):
            if e.to_gate not in seen:
                seen.add(e.to_gate)
                res.append(e.to_gate)
        return res

    def direct_distance(self, from_gate: str, to_gate: str) -> float | None:
        """Return direct edge distance, or None if not directly connected."""
        return self._all_distances.get((from_gate, to_gate))

    def shortest_distance(self, from_gate: str, to_gate: str) -> float:
        """Return shortest path distance. Returns float('inf') if unreachable."""
        return self._shortest_distances.get((from_gate, to_gate), float("inf"))

    def reachable_gates(self, from_gate: str, max_distance_m: float | None = None) -> list[str]:
        """Return all gates reachable from from_gate, optionally within max_distance_m.

        Results are sorted by distance (nearest first).
        """
        candidates = []
        for gid in self._gates:
            if gid == from_gate:
                continue
            d = self.shortest_distance(from_gate, gid)
            if d < float("inf"):
                if max_distance_m is None or d <= max_distance_m:
                    candidates.append((gid, d))
        candidates.sort(key=lambda x: x[1])
        return [gid for gid, _ in candidates]

    def max_distance(self) -> float:
        """Return the maximum finite shortest-path distance in the venue."""
        finite = [d for d in self._shortest_distances.values() if d < float("inf") and d > 0]
        return max(finite) if finite else 1.0

    def summary(self) -> dict:
        """Return a summary dict for logging/display."""
        return {
            "venue_id": self.venue_id,
            "venue_name": self.venue_name,
            "gate_count": self.gate_count,
            "gate_ids": self.gate_ids,
            "geometry_type": self._config.geometry.type,
            "radius_m": self._config.geometry.radius_m,
            "connection_count": sum(len(v) for v in self._adjacency.values()) // 2,
        }
