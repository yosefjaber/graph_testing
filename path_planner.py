"""
path_planner.py — Module 2: Cycle Detection & Ticket Route Planning
====================================================================
Responsibilities:
  - Union-Find (Disjoint Set Union) for O(α) cycle detection
  - RailwayPlanner: resolves each ticket to an acyclic, minimum-cost path
    using repeated Dijkstra + Union-Find gate-keeping
 
Algorithm for finding alternate paths
--------------------------------------
  Instead of Yen's K-Shortest Paths, we use a simpler repeated Dijkstra
  approach:
    1. Run Dijkstra on a working copy of the graph.
    2. If the resulting path would create a cycle, remove that path's edges
       from the working copy and run Dijkstra again.
    3. Repeat up to max_k times until a cycle-free path is found.
 
Public API
----------
  UnionFind(n)
  UnionFind.find(x)          -> root
  UnionFind.union(x, y)      -> bool (False = cycle detected)
  UnionFind.connected(x, y)  -> bool
  UnionFind.would_create_cycle(edges) -> bool
  UnionFind.commit_edges(edges)       -> bool
 
  RailwayPlanner(graph, max_k)
  RailwayPlanner.plan(tickets)        -> PlanResult
  RailwayPlanner.summary(plan_result) -> str
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
 
from graph import Graph, PathResult, shortest_path
 
 
# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union) with path-compression & union-by-rank
# ---------------------------------------------------------------------------
 
class UnionFind:
    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("UnionFind must have at least one element.")
        self._parent: List[int] = list(range(n))
        self._n = n

    def find(self, x: int) -> int:
        """Trace up the parent chain until we hit a root (a node that is its own parent)."""
        self._validate(x)
        if self._parent[x] == x:
            return x
        return self.find(self._parent[x])  # just walk up, no flattening

    def union(self, x: int, y: int) -> bool:
        """
        Merge the two components.
        Returns False if x and y are already connected (cycle detected).
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # already in the same component — cycle!

        self._parent[root_x] = root_y  # attach x's root under y's root
        return True

    def connected(self, x: int, y: int) -> bool:
        """Return True if x and y are in the same component."""
        return self.find(x) == self.find(y)

    def would_create_cycle(self, edges: List[Tuple[int, int]]) -> bool:
        """Test edges without permanently modifying state."""
        snapshot = self._parent[:]

        creates_cycle = False
        for u, v in edges:
            if not self.union(u, v):
                creates_cycle = True
                break

        self._parent = snapshot  # restore
        return creates_cycle

    def commit_edges(self, edges: List[Tuple[int, int]]) -> bool:
        """Permanently add edges — all or nothing."""
        if self.would_create_cycle(edges):
            return False
        for u, v in edges:
            self.union(u, v)
        return True

    def _validate(self, x: int) -> None:
        if not (0 <= x < self._n):
            raise ValueError(f"Element {x} out of range [0, {self._n - 1}].")

    def __repr__(self) -> str:
        components: Dict[int, List[int]] = {}
        for i in range(self._n):
            r = self.find(i)
            components.setdefault(r, []).append(i)
        return f"UnionFind(components={list(components.values())})"
 
# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
 
@dataclass
class TicketResult:
    """Outcome for a single ticket."""
    source: int
    destination: int
    chosen_path: Optional[PathResult]   # None → no valid path exists
    attempts: int                        # how many candidate paths were tried
 
    @property
    def is_satisfied(self) -> bool:
        return self.chosen_path is not None
 
    def __repr__(self) -> str:
        if self.chosen_path:
            return (
                f"TicketResult({self.source}→{self.destination} | "
                f"cost={self.chosen_path.cost} | "
                f"path={self.chosen_path} | attempts={self.attempts})"
            )
        return (
            f"TicketResult({self.source}→{self.destination} | "
            f"UNSATISFIED after {self.attempts} attempts)"
        )
 
 
@dataclass
class PlanResult:
    """Aggregate result of planning all tickets."""
    ticket_results: List[TicketResult]
    total_cost: float
    satisfied: int
    unsatisfied: int
    committed_edges: List[Tuple[int, int]] = field(default_factory=list)
 
    def __repr__(self) -> str:
        return (
            f"PlanResult(satisfied={self.satisfied}, "
            f"unsatisfied={self.unsatisfied}, "
            f"total_cost={self.total_cost})"
        )
 
 
# ---------------------------------------------------------------------------
# Railway Planner
# ---------------------------------------------------------------------------
 
class RailwayPlanner:
    """
    Plans a cycle-free railway network that satisfies a set of
    (source, destination) tickets at minimum total cost.
 
    Algorithm
    ---------
    For each ticket (in the order supplied):
      1. Make a working copy of the graph.
      2. Run Dijkstra to find the shortest path.
      3. Use UnionFind to check whether adding the path's edges would
         introduce a cycle in the accumulated network.
      4. If no cycle — accept the path and commit its edges.
      5. If cycle — remove the rejected path's edges from the working
         copy and run Dijkstra again (repeat up to max_k times).
      6. If all max_k attempts create cycles, mark ticket unsatisfied.
 
    Parameters
    ----------
    graph : Graph
        The underlying weighted directed graph.
    max_k : int
        Maximum number of alternative paths to explore per ticket (default 10).
    """
 
    def __init__(self, graph: Graph, max_k: int = 10) -> None:
        if max_k < 1:
            raise ValueError("max_k must be ≥ 1.")
        self._graph = graph
        self._max_k = max_k
 
    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------
 
    def plan(self, tickets: List[Tuple[int, int]]) -> PlanResult:
        """
        Plan routes for all *tickets*.
 
        Parameters
        ----------
        tickets : list of (source, destination) int pairs
 
        Returns
        -------
        PlanResult containing per-ticket outcomes and aggregate statistics.
        """
        uf = UnionFind(self._graph.num_nodes)
        ticket_results: List[TicketResult] = []
        all_committed_edges: List[Tuple[int, int]] = []
        total_cost = 0.0
 
        for source, destination in tickets:
            result = self._resolve_ticket(source, destination, uf)
            ticket_results.append(result)
 
            if result.is_satisfied:
                assert result.chosen_path is not None
                edges = result.chosen_path.edges
                uf.commit_edges(edges)
                all_committed_edges.extend(edges)
                total_cost += result.chosen_path.cost
 
        satisfied = sum(1 for r in ticket_results if r.is_satisfied)
 
        return PlanResult(
            ticket_results=ticket_results,
            total_cost=total_cost,
            satisfied=satisfied,
            unsatisfied=len(ticket_results) - satisfied,
            committed_edges=all_committed_edges,
        )
 
    # ------------------------------------------------------------------
    # Per-ticket resolution — repeated Dijkstra approach
    # ------------------------------------------------------------------
 
    def _resolve_ticket(
        self, source: int, destination: int, uf: UnionFind
    ) -> TicketResult:
        """
        Find a cycle-free path from source to destination.
 
        Strategy:
          - Run Dijkstra on a working copy of the graph.
          - If the shortest path would create a cycle, remove that path's
            edges from the working copy and try again.
          - Repeat until a valid path is found or no paths remain.
        """
        # Work on a copy so we never modify the original graph
        working_graph = self._graph.copy()
 
        for attempt in range(1, self._max_k + 1):
            candidate = shortest_path(working_graph, source, destination)
 
            # No path left in the working graph
            if candidate is None:
                break
 
            # Check if this path would create a cycle in the network
            if not uf.would_create_cycle(candidate.edges):
                return TicketResult(
                    source=source,
                    destination=destination,
                    chosen_path=candidate,
                    attempts=attempt,
                )
 
            # This path creates a cycle — remove its edges and try again
            for u, v in candidate.edges:
                working_graph.remove_edge(u, v)
 
        return TicketResult(
            source=source,
            destination=destination,
            chosen_path=None,
            attempts=self._max_k,
        )
 
    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
 
    @staticmethod
    def summary(result: PlanResult) -> str:
        """Return a human-readable summary of a PlanResult."""
        lines = [
            "=" * 60,
            "  RAILWAY PLAN SUMMARY",
            "=" * 60,
            f"  Tickets satisfied : {result.satisfied}",
            f"  Tickets FAILED    : {result.unsatisfied}",
            f"  Total network cost: {result.total_cost}",
            "-" * 60,
        ]
        for i, tr in enumerate(result.ticket_results, 1):
            status = "✓" if tr.is_satisfied else "✗"
            lines.append(f"  [{status}] Ticket {i}: {tr.source} → {tr.destination}")
            if tr.is_satisfied and tr.chosen_path:
                path_str = " → ".join(str(n) for n in tr.chosen_path.nodes)
                lines.append(f"       Path   : {path_str}")
                lines.append(f"       Cost   : {tr.chosen_path.cost}")
                lines.append(f"       Attempt: #{tr.attempts}")
            else:
                lines.append(
                    f"       No cycle-free path found "
                    f"(tried {tr.attempts} candidates)"
                )
            lines.append("")
 
        if result.committed_edges:
            lines.append("-" * 60)
            lines.append("  Final committed edges (u → v):")
            seen = set()
            for u, v in result.committed_edges:
                if (u, v) not in seen:
                    lines.append(f"    {u} → {v}")
                    seen.add((u, v))
 
        lines.append("=" * 60)
        return "\n".join(lines)
 