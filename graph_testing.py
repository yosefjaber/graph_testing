from __future__ import annotations

import math
import sys
import traceback
from typing import List, Tuple

from graph import (
    Graph,
    PathResult,
    dijkstra,
    reconstruct_path,
    shortest_path,
)
from path_planner import PlanResult, RailwayPlanner, TicketResult, UnionFind

class _TestRunner:
    """Lightweight test harness (no external dependencies)."""
 
    def __init__(self, suite_name: str) -> None:
        self.suite = suite_name
        self._passed = 0
        self._failed = 0
        self._errors: List[str] = []
 
    def assert_equal(self, actual, expected, label: str = "") -> None:
        if actual == expected:
            self._passed += 1
        else:
            self._failed += 1
            msg = f"  FAIL [{label}]: expected {expected!r}, got {actual!r}"
            self._errors.append(msg)
            print(msg)
 
    def assert_true(self, condition: bool, label: str = "") -> None:
        self.assert_equal(condition, True, label)
 
    def assert_false(self, condition: bool, label: str = "") -> None:
        self.assert_equal(condition, False, label)
 
    def assert_none(self, value, label: str = "") -> None:
        self.assert_true(value is None, label)
 
    def assert_not_none(self, value, label: str = "") -> None:
        self.assert_true(value is not None, label)
 
    def assert_almost_equal(
        self, actual: float, expected: float, tol: float = 1e-9, label: str = ""
    ) -> None:
        self.assert_true(abs(actual - expected) < tol, label)
 
    def assert_raises(self, exc_type, callable_, *args, label: str = "") -> None:
        try:
            callable_(*args)
            self._failed += 1
            msg = f"  FAIL [{label}]: expected {exc_type.__name__} but nothing raised"
            self._errors.append(msg)
            print(msg)
        except exc_type:
            self._passed += 1
        except Exception as e:
            self._failed += 1
            msg = f"  FAIL [{label}]: expected {exc_type.__name__}, got {type(e).__name__}: {e}"
            self._errors.append(msg)
            print(msg)
 
    def report(self) -> bool:
        total = self._passed + self._failed
        status = "PASSED" if self._failed == 0 else "FAILED"
        print(f"\n  [{self.suite}] {status}: {self._passed}/{total} tests passed")
        return self._failed == 0
    
def _build_sample_graph() -> Graph:
    """
    Build the sample graph
    """
    edges: List[Tuple[int, int, float]] = [
        (0, 1, 8),
        (0, 2, 4),
        (1, 2, 2),
        (3, 1, 6),
        (3, 5, 9),
        (1, 5, 11),
        (2, 6, 1),
        (2, 4, 4),
        (5, 6, 2),
        (5, 7, 3),
        (7, 6, 2),
        (6, 4, 5),
        (4, 6, 5),   # reverse edge
    ]
    return Graph.from_edge_list(edges, num_nodes=8)
 
 
def test_graph_module() -> bool:
    t = _TestRunner("graph.py")
 
    # Graph construction
    g = _build_sample_graph()
    t.assert_equal(g.num_nodes, 8, "num_nodes")
    t.assert_true(g.has_edge(0, 1), "has_edge(0,1)")
    t.assert_false(g.has_edge(1, 0), "has_edge(1,0) directed")
 
    # Invalid node
    t.assert_raises(ValueError, g.add_edge, 99, 0, 1.0, label="invalid node add_edge")
    t.assert_raises(ValueError, Graph, 0, label="empty graph")
    t.assert_raises(ValueError, g.add_edge, 0, 1, -5.0, label="negative weight")
 
    # Neighbours
    nbrs = dict(g.neighbours(0))
    t.assert_equal(nbrs[1], 8.0, "neighbour weight 0→1")
    t.assert_equal(nbrs[2], 4.0, "neighbour weight 0→2")
 
    # Graph.copy()
    g_copy = g.copy()
    t.assert_equal(g_copy.num_nodes, g.num_nodes, "copy num_nodes")
    t.assert_equal(len(g_copy.all_edges()), len(g.all_edges()), "copy edge count")
    # Modifying copy should not affect original
    g_copy.remove_edge(0, 1)
    t.assert_true(g.has_edge(0, 1), "original unaffected after copy mutation")
    t.assert_false(g_copy.has_edge(0, 1), "copy edge removed")
 
    # Dijkstra: simple triangle 0-->1-->2
    tri = Graph.from_edge_list([(0, 1, 1), (1, 2, 2), (0, 2, 10)], num_nodes=3)
    dist, prev = dijkstra(tri, source=0)
    t.assert_almost_equal(dist[2], 3.0, label="dijkstra triangle dist[2]")
    t.assert_equal(prev[2], 1, "dijkstra prev[2]")
 
    # Dijkstra: unreachable node
    dist2, _ = dijkstra(tri, source=2)
    t.assert_true(math.isinf(dist2[0]), "unreachable node is inf")
 
    # reconstruct_path
    path = reconstruct_path(prev, source=0, destination=2)
    t.assert_equal(path, [0, 1, 2], "reconstruct_path triangle")
 
    # reconstruct_path: unreachable
    _, prev2 = dijkstra(tri, source=2)
    t.assert_none(reconstruct_path(prev2, 2, 0), "reconstruct unreachable → None")
 
    # shortest_path wrapper
    sp = shortest_path(tri, 0, 2)
    t.assert_not_none(sp, "shortest_path not None")
    assert sp is not None
    t.assert_almost_equal(sp.cost, 3.0, label="shortest_path cost")
    t.assert_equal(sp.nodes, (0, 1, 2), "shortest_path nodes")
 
    # shortest_path: no path
    sp_none = shortest_path(tri, 2, 0)
    t.assert_none(sp_none, "shortest_path none")
 
    # PathResult.edges helper
    pr = PathResult(cost=5.0, nodes=(0, 1, 2, 3))
    t.assert_equal(pr.edges, [(0, 1), (1, 2), (2, 3)], "PathResult.edges")
 
    return t.report()

test_graph_module()