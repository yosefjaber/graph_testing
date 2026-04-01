"""
test_path_planner.py
Subgroup B — Unit Test Suite for path_planner.py

Run:
    pytest test_path_planner.py -v
    coverage run -m pytest test_path_planner.py && coverage report -m
"""

import pytest
from path_planner import UnionFind, RailwayPlanner
from graph import Graph


# ---------------------------------------------------------------------------
# UnionFind tests
# ---------------------------------------------------------------------------

class TestUnionFind:

    # Test 1 — Single-node self-union (edge case: n=1)
    # Input   : UnionFind(n=1), union(0, 0)
    # Expected: False  (0 and 0 share the same root; no merge happens)
    # Purpose : Verifies the cycle guard works on the smallest possible graph.
    def test_single_node_self_union_returns_false(self):
        uf = UnionFind(1)
        result = uf.union(0, 0)
        assert result is False

    # Test 2 — Triangle cycle detection via would_create_cycle
    # Input   : n=3, union(0,1), union(1,2), then would_create_cycle([(0,2)])
    # Expected: True  (adding edge 0-2 would close a triangle)
    # Purpose : Confirms cycle detection before committing edges.
    def test_would_create_cycle_detects_triangle(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.would_create_cycle([(0, 2)]) is True

    # Test 2b — would_create_cycle must NOT permanently modify state
    # Input   : Same setup as Test 2; check connectivity after the dry run.
    # Expected: 0 and 2 are NOT connected (the dry run was rolled back).
    # Purpose : Validates the snapshot-restore mechanism in would_create_cycle.
    def test_would_create_cycle_does_not_modify_state(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        uf.would_create_cycle([(1, 2)])   # dry-run only
        # 1 and 2 should still be in separate components
        assert uf.connected(1, 2) is False

    # Test 3 — commit_edges all-or-nothing rollback
    # Input   : n=3, commit_edges([(0,1), (1,2), (0,2)])
    # Expected: False  (the third edge would create a cycle; nothing committed)
    # Purpose : Verifies that commit_edges rolls back when a cycle is detected.
    def test_commit_edges_rejects_cyclic_set(self):
        uf = UnionFind(3)
        result = uf.commit_edges([(0, 1), (1, 2), (0, 2)])
        assert result is False
        # After rollback, no edges should be committed
        assert uf.connected(0, 1) is False
        assert uf.connected(1, 2) is False

    # Test 4 — Out-of-range element raises ValueError
    # Input   : UnionFind(n=2), find(5)
    # Expected: ValueError
    # Purpose : Validates the boundary-check in _validate.
    def test_out_of_range_raises_value_error(self):
        uf = UnionFind(2)
        with pytest.raises(ValueError):
            uf.find(5)

    # Test 4b — Constructor boundary: n=0 raises ValueError
    # Input   : UnionFind(0)
    # Expected: ValueError
    # Purpose : Confirms the constructor rejects degenerate sizes.
    def test_constructor_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            UnionFind(0)

    # Test — Acyclic edge set commits successfully
    # Input   : n=3, commit_edges([(0,1), (1,2)])
    # Expected: True, and 0 connected to 2 afterwards
    # Purpose : Positive path through commit_edges.
    def test_commit_edges_acyclic_set_succeeds(self):
        uf = UnionFind(3)
        result = uf.commit_edges([(0, 1), (1, 2)])
        assert result is True
        assert uf.connected(0, 2) is True


# ---------------------------------------------------------------------------
# RailwayPlanner integration tests
# ---------------------------------------------------------------------------

def _linear_graph() -> Graph:
    """0 --4--> 1 --3--> 2  (directed, weights 4 and 3)"""
    g = Graph(3)
    g.add_edge(0, 1, 4.0)
    g.add_edge(1, 2, 3.0)
    return g


def _disconnected_graph() -> Graph:
    """0 --1--> 1 ; node 2 is isolated (no edges reach it)"""
    g = Graph(3)
    g.add_edge(0, 1, 1.0)
    return g


# Test 5 — Happy-path: planner finds a path on a linear graph
# Input   : 3-node graph 0→1→2, ticket (0, 2)
# Expected: PlanResult with satisfied=1, total_cost=7.0
# Purpose : End-to-end smoke test of RailwayPlanner.plan.
#
# NOTE: This test exposes a bug in graph.py reconstruct_path (line 159):
#   `if not path or path[0] == source: return None`
# The condition should be `path[0] != source`. As written, reconstruct_path
# always returns None for any valid path, causing shortest_path to return
# None and this test to fail. Fix the `==` → `!=` in graph.py to pass.
def test_railway_planner_happy_path():
    g = _linear_graph()
    planner = RailwayPlanner(g, max_k=3)
    result = planner.plan([(0, 2)])

    assert result.satisfied == 1
    assert result.unsatisfied == 0
    assert result.total_cost == pytest.approx(7.0)
    assert result.ticket_results[0].is_satisfied is True
    assert result.ticket_results[0].chosen_path is not None


# Test 6 — No path available: disconnected graph
# Input   : Graph where node 2 is isolated, ticket (0, 2)
# Expected: PlanResult with satisfied=0, chosen_path=None
# Purpose : Confirms graceful failure when no path exists.
def test_railway_planner_disconnected_graph():
    g = _disconnected_graph()
    planner = RailwayPlanner(g, max_k=3)
    result = planner.plan([(0, 2)])

    assert result.satisfied == 0
    assert result.unsatisfied == 1
    assert result.total_cost == 0.0
    assert result.ticket_results[0].is_satisfied is False
    assert result.ticket_results[0].chosen_path is None


# Test — max_k=0 rejected by constructor
# Input   : RailwayPlanner(graph, max_k=0)
# Expected: ValueError
# Purpose : Validates constructor guard on max_k.
def test_railway_planner_invalid_max_k():
    g = _linear_graph()
    with pytest.raises(ValueError):
        RailwayPlanner(g, max_k=0)
