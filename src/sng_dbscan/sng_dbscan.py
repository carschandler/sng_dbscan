import itertools
from typing import Any

import numpy as np
from numpy.typing import NDArray


class SNG_DBSCAN:
    def __init__(
        self,
        sampling_rate: float,
        max_dist: float,
        min_points: int,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.sampling_rate = sampling_rate
        self.max_dist = max_dist
        self.min_points = min_points
        self.rng = rng

    def fit_predict(
        self, x: NDArray, similarity_measure=lambda x: np.linalg.norm(x, axis=1)
    ):
        graph = Graph(x)
        num_nodes = len(graph.nodes)
        indices = np.arange(num_nodes)
        n_sample = int(np.ceil(self.sampling_rate * num_nodes))

        for i_node, node in enumerate(graph.nodes):
            i_sample = self.rng.choice(indices, n_sample, replace=False)
            sample = graph.nodes[i_sample]
            norms = similarity_measure(node - sample, axis=1)
            indices_in_range = i_sample[norms <= self.max_dist]
            for i_in_range in indices_in_range:
                graph.add_edge(i_node, int(i_in_range))

        nodes_with_min_edges = set(
            [i for i, edge in enumerate(graph.edges) if len(edge) >= self.min_points]
        )

        connected_components: list[set[int]] = []

        # Iterate through the nodes of degree >= min_points. If a node in this
        # list has already been visited, it will be removed.
        while len(nodes_with_min_edges) > 0:
            # Start the formation of this component by adding the initial node
            # as a candidate for addition into the component
            component_candidates = set()
            # component_candidates.add(next(iter(nodes_with_min_edges)))
            component_candidates.add(nodes_with_min_edges.pop())
            component = set()
            while len(component_candidates) > 0:
                candidate = component_candidates.pop()
                component.add(candidate)
                valid_edge_nodes = graph.edges[candidate] & nodes_with_min_edges
                component_candidates.update(valid_edge_nodes)
                nodes_with_min_edges -= valid_edge_nodes
            connected_components.append(component)

        clusters = connected_components

        unclustered_nodes = set(range(num_nodes)) - set(itertools.chain(*clusters))
        for node in unclustered_nodes:
            node_in_clusters = [
                any([edge_node in cluster for edge_node in graph.edges[node]])
                for cluster in clusters
            ]
            if any(node_in_clusters):
                i_cluster = int(
                    self.rng.choice(np.arange(len(clusters))[node_in_clusters])
                )
                clusters[i_cluster].add(node)

        labels = -np.ones(num_nodes, np.int32)
        for label, cluster in enumerate(clusters):
            labels[list(cluster)] = label

        return labels


class Graph:
    def __init__(self, nodes: NDArray):
        self.nodes = nodes
        # self.edges: set[tuple[int, int]] = set()
        self.edges: list[set[int]] = [set() for _ in nodes]

    def add_edge(self, index_1: int, index_2: int):
        """Add the nodes to each other's set of connecting nodes."""
        self.edges[index_1].add(index_2)
        self.edges[index_2].add(index_1)
