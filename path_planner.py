"""
path_planner.py — Module 2: Cycle Detection & Ticket Route Planning
====================================================================
Responsibilities:
  - Union-Find (Disjoint Set Union) for O(α) cycle detection
  - RailwayPlanner: resolves each ticket to an acyclic, minimum-cost path
    using Yen's K-Shortest Paths + Union-Find gate-keeping
 
Public API
----------
  UnionFind(n)
  UnionFind.find(x)          -> root
  UnionFind.union(x, y)      -> bool (False = cycle detected)
  UnionFind.connected(x, y)  -> bool
  UnionFind.would_create_cycle(edges) -> bool
 
  RailwayPlanner(graph, max_k)
  RailwayPlanner.plan(tickets)          -> PlanResult
  RailwayPlanner.summary(plan_result)   -> str
"""
 
from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
 
from graph import Graph, PathResult, yen_k_shortest_paths
 
 
# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union) with path-compression & union-by-rank
# ---------------------------------------------------------------------------
 
class UnionFind:
    """
    Disjoint Set Union structure for *n* elements (labelled 0 … n-1).
 
    Used to detect cycles when edges are incrementally added to a graph:
    an edge (u, v) introduces a cycle iff u and v are already in the same
    connected component.
    """
 
    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError("UnionFind must have at least one element.")
        self._parent: List[int] = list(range(n))
        self._rank: List[int] = [0] * n
        self._n = n
 
    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
 
    def find(self, x: int) -> int:
        """
        Return the representative (root) of the set containing *x*.
        Uses path-compression for amortised O(α) performance.
        """
        self._validate(x)
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]
 
    def union(self, x: int, y: int) -> bool:
        """
        Merge the sets containing *x* and *y*.
 
        Returns
        -------
        True  — sets were disjoint; merge performed successfully.
        False — x and y were already in the same set (adding this edge
                would create a cycle).
        """
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False  # cycle detected
 
        # Union by rank keeps the tree shallow
        if self._rank[root_x] < self._rank[root_y]:
            root_x, root_y = root_y, root_x
        self._parent[root_y] = root_x
        if self._rank[root_x] == self._rank[root_y]:
            self._rank[root_x] += 1
        return True
 
    def connected(self, x: int, y: int) -> bool:
        """Return True if *x* and *y* belong to the same component."""
        return self.find(x) == self.find(y)
 
    # ------------------------------------------------------------------
    # Convenience: non-destructive cycle check
    # ------------------------------------------------------------------
 
    def would_create_cycle(self, edges: List[Tuple[int, int]]) -> bool:
        """
        Check whether adding *all* given (u, v) edges to the current
        structure would introduce a cycle — without permanently modifying
        the structure.
 
        Internally snapshots state, tests, then restores.
        """
        snapshot_parent = self._parent[:]
        snapshot_rank = self._rank[:]
 
        creates_cycle = False
        for u, v in edges:
            if not self.union(u, v):
                creates_cycle = True
                break
 
        # Restore
        self._parent = snapshot_parent
        self._rank = snapshot_rank
        return creates_cycle
 
    def commit_edges(self, edges: List[Tuple[int, int]]) -> bool:
        """
        Permanently add *edges* to the structure.
 
        Returns True on success, False if any edge would create a cycle
        (in which case NO edges are committed — all-or-nothing semantics).
        """
        if self.would_create_cycle(edges):
            return False
        for u, v in edges:
            self.union(u, v)
        return True
 
    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
 
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
      1. Enumerate up to *max_k* shortest paths via Yen's algorithm.
      2. Attempt each candidate path in cost order.
      3. Use UnionFind to check whether adding the path's edges would
         introduce a cycle in the accumulated network.
      4. Accept the first valid (cycle-free) path; commit its edges.
      5. If all *max_k* candidates create cycles, mark ticket unsatisfied.
 
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
 
    def plan(
        self, tickets: List[Tuple[int, int]]
    ) -> PlanResult:
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
    # Per-ticket resolution
    # ------------------------------------------------------------------
 
    def _resolve_ticket(
        self, source: int, destination: int, uf: UnionFind
    ) -> TicketResult:
        """
        Try candidate paths in ascending cost order.
        Accept the first one that does not create a cycle.
        """
        candidates = yen_k_shortest_paths(
            self._graph, source, destination, self._max_k
        )
 
        for attempt, candidate in enumerate(candidates, start=1):
            if not uf.would_create_cycle(candidate.edges):
                return TicketResult(
                    source=source,
                    destination=destination,
                    chosen_path=candidate,
                    attempts=attempt,
                )
 
        return TicketResult(
            source=source,
            destination=destination,
            chosen_path=None,
            attempts=len(candidates),
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
 