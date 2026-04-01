from __future__ import annotations
 
import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple
 
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
 
class Graph:
    """
    Weighted directed graph backed by an adjacency list.

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
 
    def copy(self) -> "Graph":
        """Return an independent copy of this graph."""
        new_graph = Graph(self._num_nodes)
        for edge in self.all_edges():
            new_graph.add_edge(edge.source, edge.destination, edge.weight)
        return new_graph
 
    @classmethod
    def from_edge_list(
        cls, edges: List[Tuple[int, int, float]], num_nodes: int
    ) -> "Graph":
        """
        Construct a Graph from a list of (source, destination, weight) tuples.

        g = Graph.from_edge_list([(0, 1, 4), (1, 2, 2)], num_nodes=3)
        """
        g = cls(num_nodes)
        for src, dst, w in edges:
            g.add_edge(src, dst, w)
        return g

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
 
    def _validate_node(self, node: int) -> None:
        if not (0 <= node < self._num_nodes):
            raise ValueError(
                f"Node {node} is out of range [0, {self._num_nodes - 1}]."
            )
 
    def __repr__(self) -> str:
        return f"Graph(nodes={self._num_nodes}, edges={len(self.all_edges())})"
 
def dijkstra(
    graph: Graph, source: int
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
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
 
    if not path or path[0] == source:
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
 