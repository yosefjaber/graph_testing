"""
main.py — Entry point & Test Suite
===================================
Demonstrates and stress-tests both modules:
  Module 1 — graph.py       : Graph, dijkstra, shortest_path
  Module 2 — path_planner.py: UnionFind, RailwayPlanner
 
Run
---
    python main.py              # runs demo + all tests
    python main.py --demo-only  # runs only the worked example
    python main.py --test-only  # runs only the test suite
"""
 
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

 
# ===========================================================================
# Helpers
# ===========================================================================
 
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
 
 
# ===========================================================================
# Module 1 tests — graph.py
# ===========================================================================
 
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
 
    # --- Graph construction ---
    g = _build_sample_graph()
    t.assert_equal(g.num_nodes, 8, "num_nodes")
    t.assert_true(g.has_edge(0, 1), "has_edge(0,1)")
    t.assert_false(g.has_edge(1, 0), "has_edge(1,0) directed")
 
    # --- Invalid node ---
    t.assert_raises(ValueError, g.add_edge, 99, 0, 1.0, label="invalid node add_edge")
    t.assert_raises(ValueError, Graph, 0, label="empty graph")
    t.assert_raises(ValueError, g.add_edge, 0, 1, -5.0, label="negative weight")
 
    # --- Neighbours ---
    nbrs = dict(g.neighbours(0))
    t.assert_equal(nbrs[1], 8.0, "neighbour weight 0→1")
    t.assert_equal(nbrs[2], 4.0, "neighbour weight 0→2")
 
    # --- Graph.copy() ---
    g_copy = g.copy()
    t.assert_equal(g_copy.num_nodes, g.num_nodes, "copy num_nodes")
    t.assert_equal(len(g_copy.all_edges()), len(g.all_edges()), "copy edge count")
    # Modifying copy should not affect original
    g_copy.remove_edge(0, 1)
    t.assert_true(g.has_edge(0, 1), "original unaffected after copy mutation")
    t.assert_false(g_copy.has_edge(0, 1), "copy edge removed")
 
    # --- Dijkstra: simple triangle 0→1→2 ---
    tri = Graph.from_edge_list([(0, 1, 1), (1, 2, 2), (0, 2, 10)], num_nodes=3)
    dist, prev = dijkstra(tri, source=0)
    t.assert_almost_equal(dist[2], 3.0, label="dijkstra triangle dist[2]")
    t.assert_equal(prev[2], 1, "dijkstra prev[2]")
 
    # --- Dijkstra: unreachable node ---
    dist2, _ = dijkstra(tri, source=2)
    t.assert_true(math.isinf(dist2[0]), "unreachable node is inf")
 
    # --- reconstruct_path ---
    path = reconstruct_path(prev, source=0, destination=2)
    t.assert_equal(path, [0, 1, 2], "reconstruct_path triangle")
 
    # --- reconstruct_path: unreachable ---
    _, prev2 = dijkstra(tri, source=2)
    t.assert_none(reconstruct_path(prev2, 2, 0), "reconstruct unreachable → None")
 
    # --- shortest_path wrapper ---
    sp = shortest_path(tri, 0, 2)
    t.assert_not_none(sp, "shortest_path not None")
    assert sp is not None
    t.assert_almost_equal(sp.cost, 3.0, label="shortest_path cost")
    t.assert_equal(sp.nodes, (0, 1, 2), "shortest_path nodes")
 
    # --- shortest_path: no path ---
    sp_none = shortest_path(tri, 2, 0)
    t.assert_none(sp_none, "shortest_path none")
 
    # --- PathResult.edges helper ---
    pr = PathResult(cost=5.0, nodes=(0, 1, 2, 3))
    t.assert_equal(pr.edges, [(0, 1), (1, 2), (2, 3)], "PathResult.edges")
 
    return t.report()
 
 
# ===========================================================================
# Module 2 tests — path_planner.py
# ===========================================================================
 
def test_path_planner_module() -> bool:
    t = _TestRunner("path_planner.py")
 
    # ----------------------------------------------------------------
    # UnionFind — basic
    # ----------------------------------------------------------------
    uf = UnionFind(5)
    t.assert_false(uf.connected(0, 1), "initially disconnected")
    merged = uf.union(0, 1)
    t.assert_true(merged, "union returns True on first merge")
    t.assert_true(uf.connected(0, 1), "connected after union")
 
    # Cycle detection: union already-connected elements
    cycle = uf.union(0, 1)
    t.assert_false(cycle, "union returns False on duplicate → cycle")
 
    # Path compression / transitivity
    uf2 = UnionFind(4)
    uf2.union(0, 1)
    uf2.union(1, 2)
    t.assert_true(uf2.connected(0, 2), "transitivity 0-1-2")
    t.assert_false(uf2.connected(0, 3), "node 3 isolated")
 
    # ----------------------------------------------------------------
    # UnionFind — would_create_cycle (non-destructive)
    # ----------------------------------------------------------------
    uf3 = UnionFind(4)
    uf3.union(0, 1)
    uf3.union(2, 3)
 
    # Should NOT create cycle: 1→2 bridges two components
    t.assert_false(
        uf3.would_create_cycle([(1, 2)]),
        "would_create_cycle: bridging — False",
    )
    # Verify uf3 was NOT modified
    t.assert_false(uf3.connected(1, 2), "would_create_cycle: non-destructive")
 
    # WOULD create cycle: 0→1 already joined
    t.assert_true(
        uf3.would_create_cycle([(0, 1)]),
        "would_create_cycle: existing edge — True",
    )
 
    # ----------------------------------------------------------------
    # UnionFind — commit_edges (all-or-nothing)
    # ----------------------------------------------------------------
    uf4 = UnionFind(4)
    success = uf4.commit_edges([(0, 1), (1, 2)])
    t.assert_true(success, "commit_edges success")
    t.assert_true(uf4.connected(0, 2), "committed transitivity")
 
    # Attempt commit that would create cycle — should fail and not modify
    fail = uf4.commit_edges([(0, 2)])
    t.assert_false(fail, "commit_edges cycle → False")
    t.assert_true(uf4.connected(0, 2), "state unchanged after failed commit")
 
    # ----------------------------------------------------------------
    # UnionFind — validation
    # ----------------------------------------------------------------
    t.assert_raises(ValueError, UnionFind, 0, label="UnionFind(0)")
    uf5 = UnionFind(3)
    t.assert_raises(ValueError, uf5.find, 5, label="out-of-range find")
 
    # ----------------------------------------------------------------
    # RailwayPlanner — basic: two disjoint tickets
    # ----------------------------------------------------------------
    g = _build_sample_graph()
    planner = RailwayPlanner(g, max_k=5)
 
    tickets: List[Tuple[int, int]] = [(0, 4), (3, 6)]
    result = planner.plan(tickets)
 
    t.assert_equal(result.satisfied, 2, "both tickets satisfied")
    t.assert_equal(result.unsatisfied, 0, "no unsatisfied tickets")
    t.assert_true(result.total_cost > 0, "positive total cost")
    t.assert_equal(len(result.ticket_results), 2, "two ticket results")
    for tr in result.ticket_results:
        t.assert_true(tr.is_satisfied, f"ticket {tr.source}→{tr.destination} satisfied")
 
    # ----------------------------------------------------------------
    # RailwayPlanner — cycle avoidance
    # ----------------------------------------------------------------
    chain = Graph.from_edge_list(
        [(0, 1, 1), (1, 2, 1), (2, 3, 1)], num_nodes=4
    )
    chain_planner = RailwayPlanner(chain, max_k=5)
    chain_result = chain_planner.plan([(0, 3), (3, 0)])
    t.assert_equal(chain_result.satisfied, 1, "chain: only 1 satisfiable")
    t.assert_equal(chain_result.unsatisfied, 1, "chain: 1 unsatisfied (3→0 impossible)")
 
    # ----------------------------------------------------------------
    # RailwayPlanner — single-node (source == destination)
    # ----------------------------------------------------------------
    trivial = Graph.from_edge_list([(0, 1, 1)], num_nodes=2)
    triv_planner = RailwayPlanner(trivial, max_k=3)
    triv_result = triv_planner.plan([(0, 0)])
    t.assert_equal(triv_result.satisfied, 1, "trivial same-node ticket satisfied")
 
    # ----------------------------------------------------------------
    # RailwayPlanner — empty ticket list
    # ----------------------------------------------------------------
    empty_result = planner.plan([])
    t.assert_equal(empty_result.satisfied, 0, "empty tickets → 0 satisfied")
    t.assert_equal(empty_result.total_cost, 0.0, "empty tickets → cost 0")
 
    # ----------------------------------------------------------------
    # RailwayPlanner — forced fallback via repeated Dijkstra
    # ----------------------------------------------------------------
    # Graph:
    #   0→1 (1), 1→2 (1)          ← cheapest path for ticket (0,2) costs 2
    #   0→2 (5)                    ← direct, expensive
    #   1→3 (4), 2→3 (1)          ← paths to node 3
    #
    # Ticket 1: (0, 2)  → committed: 0→1→2  (edges 0-1, 1-2)
    # Ticket 2: (1, 3)
    #   Attempt 1: 1→2→3 (cost 2) → would create cycle (1 & 2 already connected)
    #              edges 1→2 and 2→3 removed from working graph
    #   Attempt 2: 1→3   (cost 4) → no cycle ✓
    fallback_g = Graph.from_edge_list(
        [(0, 1, 1), (1, 2, 1), (0, 2, 5), (2, 3, 1), (1, 3, 4)],
        num_nodes=4,
    )
    fb_planner = RailwayPlanner(fallback_g, max_k=5)
    fb_result = fb_planner.plan([(0, 2), (1, 3)])
 
    # Ticket 1: cheapest path 0→1→2 (cost 2)
    t.assert_true(fb_result.ticket_results[0].is_satisfied, "fallback ticket 1 sat")
    t.assert_almost_equal(
        fb_result.ticket_results[0].chosen_path.cost, 2.0,  # type: ignore[union-attr]
        label="fallback ticket 1 cost",
    )
    # Ticket 2: first candidate 1→2→3 creates cycle; fallback to 1→3 (cost 4)
    t.assert_true(fb_result.ticket_results[1].is_satisfied, "fallback ticket 2 sat")
    t.assert_almost_equal(
        fb_result.ticket_results[1].chosen_path.cost, 4.0,  # type: ignore[union-attr]
        label="fallback ticket 2 cost (repeated Dijkstra used)",
    )
    t.assert_true(
        fb_result.ticket_results[1].attempts >= 2,
        "fallback ticket 2 needed ≥2 attempts",
    )
 
    return t.report()
 
 
# ===========================================================================
# Worked demo — mirrors the problem-image graph
# ===========================================================================
 
def run_demo() -> None:
    print("\n" + "=" * 60)
    print("  TICKET TO RIDE — WORKED EXAMPLE")
    print("=" * 60)
 
    g = _build_sample_graph()
    print(f"\n  Graph: {g}")
 
    tickets: List[Tuple[int, int]] = [
        (3, 4),   # Ticket 1: city 3 → city 4
        (0, 7),   # Ticket 2: city 0 → city 7
        (1, 4),   # Ticket 3: city 1 → city 4
    ]
 
    print("\n  Tickets to resolve:")
    for i, (s, d) in enumerate(tickets, 1):
        print(f"    {i}. city {s} → city {d}")
 
    planner = RailwayPlanner(g, max_k=10)
    result = planner.plan(tickets)
 
    print()
    print(RailwayPlanner.summary(result))
 
 
# ===========================================================================
# Main
# ===========================================================================
 
def main() -> None:
    args = sys.argv[1:]
    demo_only = "--demo-only" in args
    test_only = "--test-only" in args
 
    all_passed = True
 
    if not demo_only:
        print("\n━━━  MODULE 1 TESTS (graph.py)  ━━━")
        passed1 = test_graph_module()

        print("\n━━━  MODULE 2 TESTS (path_planner.py)  ━━━")
        passed2 = test_path_planner_module()

        all_passed = passed1 and passed2

        if all_passed:
            print("\n  ✓  All tests passed.\n")
        else:
            print("\n  ✗  Some tests FAILED — see above.\n")

    if not test_only:
        run_demo()

    sys.exit(0 if all_passed else 1)

 
if __name__ == "__main__":
    main()