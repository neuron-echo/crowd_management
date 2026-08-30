"""Tests for venue configuration, topology, and loader.

Tests cover:
1. Venue config loading from YAML
2. Circular gate topology
3. Camera → gate mapping
4. Auto-derived circular connections
5. Explicit connections
6. Parametric N-gate support
7. Distance computation (Floyd-Warshall)
8. Validation errors
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import yaml

from crowd_management.venue.loader import load_venue_config, load_venue_topology
from crowd_management.venue.models import VenueConfig, GateConfig, GeometryConfig, ConnectionConfig
from crowd_management.venue.topology import VenueTopology


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def four_gate_config() -> VenueConfig:
    """Create a 4-gate circular venue configuration."""
    return VenueConfig(
        id="test_venue",
        name="Test Circular Venue",
        geometry=GeometryConfig(type="circular", radius_m=100.0),
        gates=[
            GateConfig(gate_id="G1", capacity=500, position_index=0, camera_id="cam_01"),
            GateConfig(gate_id="G2", capacity=600, position_index=1, camera_id="cam_02"),
            GateConfig(gate_id="G3", capacity=450, position_index=2, camera_id="cam_03"),
            GateConfig(gate_id="G4", capacity=700, position_index=3, camera_id="cam_04"),
        ],
        auto_circular_connections=True,
    )


@pytest.fixture
def four_gate_topology(four_gate_config: VenueConfig) -> VenueTopology:
    return VenueTopology(four_gate_config)


@pytest.fixture
def venue_yaml_path(tmp_path: Path) -> Path:
    """Create a temporary venue YAML file."""
    data = {
        "venue": {
            "id": "yaml_test",
            "name": "YAML Test Venue",
            "geometry": {"type": "circular", "radius_m": 50},
            "auto_circular_connections": True,
            "gates": [
                {"gate_id": "G1", "capacity": 300, "position_index": 0, "camera_id": "cam_01"},
                {"gate_id": "G2", "capacity": 400, "position_index": 1, "camera_id": "cam_02"},
                {"gate_id": "G3", "capacity": 350, "position_index": 2},
            ],
        }
    }
    yaml_path = tmp_path / "test_venue.yaml"
    with yaml_path.open("w") as f:
        yaml.dump(data, f)
    return yaml_path


# ===================================================================
# Test: Venue Config Loading
# ===================================================================

class TestVenueConfigLoading:

    def test_load_from_yaml(self, venue_yaml_path: Path):
        config = load_venue_config(venue_yaml_path)
        assert config.id == "yaml_test"
        assert config.name == "YAML Test Venue"
        assert len(config.gates) == 3
        assert config.geometry.radius_m == 50

    def test_load_topology(self, venue_yaml_path: Path):
        topo = load_venue_topology(venue_yaml_path)
        assert topo.gate_count == 3
        assert topo.venue_name == "YAML Test Venue"

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_venue_config(tmp_path / "nonexistent.yaml")

    def test_invalid_config_missing_gates(self, tmp_path: Path):
        data = {"venue": {"id": "bad", "name": "Bad", "gates": []}}
        yaml_path = tmp_path / "bad.yaml"
        with yaml_path.open("w") as f:
            yaml.dump(data, f)
        with pytest.raises(ValueError):
            load_venue_config(yaml_path)


# ===================================================================
# Test: Circular Gate Topology
# ===================================================================

class TestCircularTopology:

    def test_gate_count(self, four_gate_topology: VenueTopology):
        assert four_gate_topology.gate_count == 4

    def test_gate_ids_ordered(self, four_gate_topology: VenueTopology):
        assert four_gate_topology.gate_ids == ["G1", "G2", "G3", "G4"]

    def test_auto_circular_connections(self, four_gate_config: VenueConfig):
        conns = four_gate_config.derive_circular_connections()
        assert len(conns) == 4
        from_to = [(c.from_gate, c.to_gate) for c in conns]
        assert ("G1", "G2") in from_to
        assert ("G2", "G3") in from_to
        assert ("G3", "G4") in from_to
        assert ("G4", "G1") in from_to

    def test_arc_length(self, four_gate_config: VenueConfig):
        conns = four_gate_config.derive_circular_connections()
        expected = 100.0 * (2 * math.pi / 4)  # ~157.08
        for c in conns:
            assert abs(c.distance_m - round(expected, 2)) < 0.01

    def test_neighbors(self, four_gate_topology: VenueTopology):
        neighbors_g1 = set(four_gate_topology.neighbors("G1"))
        assert "G2" in neighbors_g1
        assert "G4" in neighbors_g1

    def test_all_reachable(self, four_gate_topology: VenueTopology):
        reachable = four_gate_topology.reachable_gates("G1")
        assert set(reachable) == {"G2", "G3", "G4"}

    def test_shortest_distance_adjacent(self, four_gate_topology: VenueTopology):
        d = four_gate_topology.shortest_distance("G1", "G2")
        expected = 100.0 * (2 * math.pi / 4)
        assert abs(d - expected) < 0.1

    def test_shortest_distance_opposite(self, four_gate_topology: VenueTopology):
        d = four_gate_topology.shortest_distance("G1", "G3")
        arc = 100.0 * (2 * math.pi / 4)
        assert abs(d - 2 * arc) < 0.1


# ===================================================================
# Test: Camera → Gate Mapping
# ===================================================================

class TestCameraGateMapping:

    def test_camera_to_gate(self, four_gate_config: VenueConfig):
        mapping = four_gate_config.get_camera_to_gate_map()
        assert mapping["cam_01"] == "G1"
        assert mapping["cam_02"] == "G2"
        assert mapping["cam_03"] == "G3"
        assert mapping["cam_04"] == "G4"

    def test_gate_to_camera(self, four_gate_config: VenueConfig):
        mapping = four_gate_config.get_gate_to_camera_map()
        assert mapping["G1"] == "cam_01"
        assert mapping["G4"] == "cam_04"

    def test_topology_camera_lookup(self, four_gate_topology: VenueTopology):
        assert four_gate_topology.gate_for_camera("cam_01") == "G1"
        assert four_gate_topology.gate_for_camera("cam_99") is None

    def test_topology_gate_camera_lookup(self, four_gate_topology: VenueTopology):
        assert four_gate_topology.camera_for_gate("G1") == "cam_01"

    def test_unmonitored_gate(self):
        config = VenueConfig(
            id="t", name="T",
            gates=[
                GateConfig(gate_id="G1", capacity=100, position_index=0, camera_id="cam_01"),
                GateConfig(gate_id="G2", capacity=100, position_index=1),
            ],
        )
        mapping = config.get_camera_to_gate_map()
        assert "G2" not in mapping.values() or mapping.get(None) is None
        gate_map = config.get_gate_to_camera_map()
        assert gate_map["G2"] is None


# ===================================================================
# Test: N-Gate Parametric Support
# ===================================================================

class TestNGateSupport:

    @pytest.mark.parametrize("n_gates", [2, 3, 4, 6, 8, 10])
    def test_n_gate_topology(self, n_gates: int):
        gates = [
            GateConfig(gate_id=f"G{i+1}", capacity=500, position_index=i)
            for i in range(n_gates)
        ]
        config = VenueConfig(id="n_gate", name="N-Gate Test", gates=gates)
        topo = VenueTopology(config)

        assert topo.gate_count == n_gates
        assert len(topo.gate_ids) == n_gates

        # In 2-gate topology, each gate has 1 neighbor (the other gate); for N >= 3, each gate has 2 neighbors
        expected_neighbors = 1 if n_gates == 2 else 2
        for gid in topo.gate_ids:
            assert len(topo.neighbors(gid)) == expected_neighbors

        # All other gates reachable from any gate
        for gid in topo.gate_ids:
            reachable = topo.reachable_gates(gid)
            assert len(reachable) == n_gates - 1

    def test_two_gate_circular_venue(self):
        """Verify exact 2-gate circular venue behavior."""
        config = VenueConfig(
            id="two_gate", name="2-Gate Venue",
            geometry=GeometryConfig(radius_m=100.0),
            gates=[
                GateConfig(gate_id="G1", capacity=3, position_index=0, camera_id="cam_01"),
                GateConfig(gate_id="G2", capacity=50, position_index=1, camera_id="cam_02"),
            ],
            auto_circular_connections=True,
        )
        topo = VenueTopology(config)
        assert topo.gate_count == 2
        assert topo.gate_ids == ["G1", "G2"]
        assert topo.neighbors("G1") == ["G2"]
        assert topo.neighbors("G2") == ["G1"]
        assert topo.reachable_gates("G1") == ["G2"]
        assert topo.gate_for_camera("cam_01") == "G1"
        assert topo.gate_for_camera("cam_02") == "G2"
        # Distance should be half circumference = pi * 100 ~ 314.16m
        d = topo.shortest_distance("G1", "G2")
        assert abs(d - math.pi * 100.0) < 0.1


# ===================================================================
# Test: Explicit Connections
# ===================================================================

class TestExplicitConnections:

    def test_explicit_overrides_auto(self):
        config = VenueConfig(
            id="explicit_test", name="Explicit",
            gates=[
                GateConfig(gate_id="G1", capacity=100, position_index=0),
                GateConfig(gate_id="G2", capacity=100, position_index=1),
                GateConfig(gate_id="G3", capacity=100, position_index=2),
            ],
            connections=[
                ConnectionConfig(**{"from": "G1", "to": "G2", "distance_m": 10.0}),
                ConnectionConfig(**{"from": "G2", "to": "G3", "distance_m": 20.0}),
            ],
        )
        effective = config.get_effective_connections()
        assert len(effective) == 2  # Explicit, not auto-derived 3

    def test_duplicate_gate_ids_rejected(self):
        with pytest.raises(ValueError, match="Duplicate gate IDs"):
            VenueConfig(
                id="bad", name="Bad",
                gates=[
                    GateConfig(gate_id="G1", capacity=100, position_index=0),
                    GateConfig(gate_id="G1", capacity=200, position_index=1),
                ],
            )

    def test_self_connection_rejected(self):
        with pytest.raises(ValueError, match="Self-connection"):
            VenueConfig(
                id="bad", name="Bad",
                gates=[
                    GateConfig(gate_id="G1", capacity=100, position_index=0),
                    GateConfig(gate_id="G2", capacity=100, position_index=1),
                ],
                connections=[
                    ConnectionConfig(**{"from": "G1", "to": "G1", "distance_m": 10.0}),
                ],
            )

    def test_gate_capacity_lookup(self, four_gate_topology: VenueTopology):
        assert four_gate_topology.get_gate_capacity("G1") == 500
        assert four_gate_topology.get_gate_capacity("G4") == 700
