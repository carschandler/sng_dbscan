from typing import Any

import numpy as np
from numpy.typing import NDArray


class SNG_DBSCAN:
    def __init__(
        self,
        x: NDArray,
        sampling_rate: float,
        max_dist: float,
        min_points: int,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.graph = Graph(x)
        self.sampling_rate = sampling_rate
        self.max_dist = max_dist
        self.min_points = min_points
        self.rng = rng

    def fit(self):
        n = len(self.graph.nodes)
        indices = np.arange(n)
        n_sample = int(np.ceil(self.sampling_rate * n))

        for i, node in enumerate(self.graph.nodes):
            i_sample = self.rng.choice(indices, n_sample)
            sample = self.graph.nodes[i_sample]
            norms = np.abs(node - sample)
            matching_indices = i_sample[norms <= self.max_dist]
            for j in matching_indices:
                self.graph.add_edge(i, j)

            min_points_mask = np.vectorize(lambda x: x >= self.min_points)(
                self.graph.edges
            )


class Graph:
    def __init__(self, nodes: NDArray):
        self.nodes = nodes
        # self.edges: set[tuple[int, int]] = set()
        self.edges: NDArray[np.object_] = np.full(len(nodes), set())

    # def add_edge(self, index_1: int, index_2: int):
    #     """Add edges in ascending index order to ensure no duplicate edges."""
    #     if index_1 < index_2:
    #         self.edges.add((index_1, index_2))
    #     else:
    #         self.edges.add((index_2, index_1))

    def add_edge(self, index_1: int, index_2: int):
        """Add the nodes to each other's set of connecting nodes."""
        self.edges[index_1].add(index_2)
        self.edges[index_2].add(index_1)
