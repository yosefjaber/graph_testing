"""
graph.py — Module 1: Graph Representation & Pathfinding
========================================================
Responsibilities:
  - Represent a weighted directed graph (Edge, Graph)
  - Dijkstra's single-source shortest-path algorithm
  - Yen's K-Shortest Paths for enumerating fallback routes
 
Public API
----------
  Graph.add_edge(u, v, weight)
  Graph.from_edge_list(edges, num_nodes)
  dijkstra(graph, source)            -> (dist, prev)
  reconstruct_path(prev, source, dest) -> list[int] | None
  yen_k_shortest_paths(graph, src, dst, k) -> list[PathResult]
"""
 
from __future__ import annotations
 
import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple
 
 
# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
 
@dataclass(frozen=True)
class Edge:
    """A single directed, weighted edge."""
    source: int
    destination: int
    weight: float
 
    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"Negative edge weight not supported: {self}")
 
 
@dataclass(frozen=True, order=True)
class PathResult:
    """A complete path with its total cost (orderable by cost)."""
    cost: float
    nodes: Tuple[int, ...] = field(compare=False)
 
    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Return the (u, v) edge pairs along this path."""
        return list(zip(self.nodes, self.nodes[1:]))
 
    def __repr__(self) -> str:
        arrow = " → ".join(str(n) for n in self.nodes)
        return f"PathResult(cost={self.cost}, path={arrow})"
 
 
# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
 
class Graph:
    """
    Weighted directed graph backed by an adjacency list.
 
    Parameters
    ----------
    num_nodes : int
        Number of nodes (labelled 0 … num_nodes-1).
    """
 
    def __init__(self, num_nodes: int) -> None:
        if num_nodes < 1:
            raise ValueError("Graph must have at least one node.")
        self._num_nodes: int = num_nodes
        # adjacency list: node -> list of (neighbour, weight)
        self._adj: Dict[int, List[Tuple[int, float]]] = {
            i: [] for i in range(num_nodes)
        }
 
    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
 
    def add_edge(self, source: int, destination: int, weight: float) -> None:
        """Add a directed edge source → destination with the given weight."""
        self._validate_node(source)
        self._validate_node(destination)
        if weight < 0:
            raise ValueError("Negative weights are not supported.")
        self._adj[source].append((destination, weight))
 
    def remove_edge(self, source: int, destination: int) -> None:
        """Remove *all* directed edges from source to destination."""
        self._adj[source] = [
            (d, w) for d, w in self._adj[source] if d != destination
        ]
 
    @classmethod
    def from_edge_list(
        cls, edges: List[Tuple[int, int, float]], num_nodes: int
    ) -> "Graph":
        """
        Construct a Graph from a list of (source, destination, weight) tuples.
 
        Example
        -------
        >>> g = Graph.from_edge_list([(0, 1, 4), (1, 2, 2)], num_nodes=3)
        """
        g = cls(num_nodes)
        for src, dst, w in edges:
            g.add_edge(src, dst, w)
        return g
 
    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
 
    @property
    def num_nodes(self) -> int:
        return self._num_nodes
 
    def neighbours(self, node: int) -> Iterator[Tuple[int, float]]:
        """Yield (neighbour, weight) for every outgoing edge from node."""
        self._validate_node(node)
        yield from self._adj[node]
 
    def has_edge(self, source: int, destination: int) -> bool:
        return any(d == destination for d, _ in self._adj[source])
 
    def all_edges(self) -> List[Edge]:
        return [
            Edge(src, dst, w)
            for src, neighbours in self._adj.items()
            for dst, w in neighbours
        ]
 
    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
 
    def _validate_node(self, node: int) -> None:
        if not (0 <= node < self._num_nodes):
            raise ValueError(
                f"Node {node} is out of range [0, {self._num_nodes - 1}]."
            )
 
    def __repr__(self) -> str:
        return f"Graph(nodes={self._num_nodes}, edges={len(self.all_edges())})"
 
 
# ---------------------------------------------------------------------------
# Dijkstra
# ---------------------------------------------------------------------------
 
def dijkstra(
    graph: Graph, source: int
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """
    Run Dijkstra's algorithm from *source* on *graph*.
 
    Returns
    -------
    dist : dict[node -> float]
        Shortest distance from source to every reachable node.
        Unreachable nodes map to math.inf.
    prev : dict[node -> int | None]
        Predecessor map for path reconstruction.
        prev[source] = None; unreachable nodes = None.
    """
    dist: Dict[int, float] = {n: math.inf for n in range(graph.num_nodes)}
    prev: Dict[int, Optional[int]] = {n: None for n in range(graph.num_nodes)}
 
    dist[source] = 0.0
    # min-heap: (tentative_distance, node)
    heap: List[Tuple[float, int]] = [(0.0, source)]
 
    while heap:
        current_dist, u = heapq.heappop(heap)
 
        # Stale entry — skip
        if current_dist > dist[u]:
            continue
 
        for v, weight in graph.neighbours(u):
            new_dist = dist[u] + weight
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))
 
    return dist, prev
 
 
def reconstruct_path(
    prev: Dict[int, Optional[int]], source: int, destination: int
) -> Optional[List[int]]:
    """
    Walk *prev* backwards from *destination* to *source*.
 
    Returns the node list [source, …, destination], or None if unreachable.
    """
    path: List[int] = []
    current: Optional[int] = destination
 
    while current is not None:
        path.append(current)
        current = prev[current]
 
    path.reverse()
 
    if not path or path[0] != source:
        return None  # destination is unreachable from source
    return path
 
 
def shortest_path(
    graph: Graph, source: int, destination: int
) -> Optional[PathResult]:
    """
    Return the single shortest PathResult from source → destination,
    or None if no path exists.
    """
    dist, prev = dijkstra(graph, source)
    if math.isinf(dist[destination]):
        return None
    nodes = reconstruct_path(prev, source, destination)
    if nodes is None:
        return None
    return PathResult(cost=dist[destination], nodes=tuple(nodes))
 
 
# ---------------------------------------------------------------------------
# Yen's K-Shortest Paths
# ---------------------------------------------------------------------------
 
def yen_k_shortest_paths(
    graph: Graph, source: int, destination: int, k: int
) -> List[PathResult]:
    """
    Yen's algorithm: enumerate up to *k* simple shortest paths from
    source → destination in ascending order of cost.
 
    Returns
    -------
    List of PathResult, length ≤ k.  Empty if no path exists at all.
    """
    if k < 1:
        raise ValueError("k must be ≥ 1.")
 
    first = shortest_path(graph, source, destination)
    if first is None:
        return []
 
    confirmed: List[PathResult] = [first]
    # Candidate heap: (cost, path_as_tuple)
    candidates: List[Tuple[float, Tuple[int, ...]]] = []
    candidate_set: set = set()  # avoid duplicates
 
    for _ in range(k - 1):
        last_path = confirmed[-1]
 
        for i in range(len(last_path.nodes) - 1):
            spur_node = last_path.nodes[i]
            root_path = last_path.nodes[: i + 1]
 
            # Build a modified graph:
            # 1. Remove edges used by confirmed paths that share root_path
            # 2. Remove root_path nodes (except spur_node) to prevent revisits
            temp_graph = _build_spur_graph(
                graph, confirmed, root_path, spur_node
            )
 
            spur_result = shortest_path(temp_graph, spur_node, destination)
            if spur_result is None:
                continue
 
            full_nodes = root_path + spur_result.nodes[1:]
            full_cost = _path_cost(graph, full_nodes)
 
            candidate_key = full_nodes
            if candidate_key not in candidate_set:
                candidate_set.add(candidate_key)
                heapq.heappush(candidates, (full_cost, full_nodes))
 
        if not candidates:
            break
 
        best_cost, best_nodes = heapq.heappop(candidates)
        confirmed.append(PathResult(cost=best_cost, nodes=best_nodes))
 
    return confirmed
 
 
# ------------------------------------------------------------------
# Yen's helpers
# ------------------------------------------------------------------
 
def _build_spur_graph(
    graph: Graph,
    confirmed: List[PathResult],
    root_path: Tuple[int, ...],
    spur_node: int,
) -> Graph:
    """
    Return a copy of *graph* with:
      - Edges removed that overlap with already-confirmed paths at root_path.
      - Root-path nodes (except spur_node) removed to avoid revisiting.
    """
    temp = Graph(graph.num_nodes)
 
    # Collect edges to block
    blocked_edges: set = set()
    for confirmed_path in confirmed:
        cp = confirmed_path.nodes
        if len(cp) > len(root_path) and cp[: len(root_path)] == root_path:
            # Block the next edge after root_path
            blocked_edges.add((cp[len(root_path) - 1], cp[len(root_path)]))
 
    # Nodes to block (all root nodes except spur_node itself)
    blocked_nodes = set(root_path[:-1])
 
    for edge in graph.all_edges():
        if edge.source in blocked_nodes or edge.destination in blocked_nodes:
            continue
        if (edge.source, edge.destination) in blocked_edges:
            continue
        temp.add_edge(edge.source, edge.destination, edge.weight)
 
    return temp
 
 
def _path_cost(graph: Graph, nodes: Tuple[int, ...]) -> float:
    """Sum edge weights along a node sequence."""
    total = 0.0
    adj_lookup: Dict[int, Dict[int, float]] = {}
    for edge in graph.all_edges():
        adj_lookup.setdefault(edge.source, {})[edge.destination] = edge.weight
 
    for u, v in zip(nodes, nodes[1:]):
        w = adj_lookup.get(u, {}).get(v, math.inf)
        total += w
    return total
 