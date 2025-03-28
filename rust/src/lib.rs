use std::collections::{BTreeSet, HashSet};

use ndarray_linalg::Norm;
use numpy::{ndarray::prelude::*, IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rand::prelude::*;

#[pyclass(get_all, set_all)]
struct SNG_DBSCAN {
    sampling_rate: f64,
    max_dist: f64,
    min_points: usize,
}

impl SNG_DBSCAN {
    fn fit_predict(&self, x: ArrayViewD<f64>) -> Array1<i32> {
        let mut graph = Graph::new(x);
        let num_nodes = graph.nodes.len();
        let indices: Vec<usize> = (0usize..num_nodes).collect();
        let n_sample = self.sampling_rate.ceil() as usize;

        let mut rng = rand::rng();

        let mut new_edges = Vec::new();

        for (i_node, node) in graph.nodes.outer_iter().enumerate() {
            let i_sample = indices
                .clone()
                .into_iter()
                .choose_multiple(&mut rng, n_sample);
            let sample = graph.nodes.select(Axis(0), &i_sample[..]);

            let mut norms = &node - &sample.view();
            norms = norms.map_axis(Axis(0), |x| x.norm());

            let indices_in_range: Vec<usize> = i_sample
                .iter()
                .zip(norms)
                .filter(|(_, norm)| *norm < self.max_dist)
                .map(|(&i, _)| i)
                .collect();

            for i_in_range in indices_in_range {
                new_edges.push((i_node, i_in_range));
            }
        }

        for (i_node, i_in_range) in new_edges {
            graph.add_edge(i_node, i_in_range);
        }

        let mut nodes_with_min_edges: BTreeSet<usize> = BTreeSet::from_iter(
            graph
                .edges
                .iter()
                .enumerate()
                .filter(|(_, edge)| edge.len() >= self.min_points)
                .map(|(i, _)| i),
        );

        let mut connected_components: Vec<HashSet<usize>> = Vec::new();

        while !nodes_with_min_edges.is_empty() {
            let mut component_candidates: BTreeSet<usize> = BTreeSet::new();
            component_candidates.insert(
                nodes_with_min_edges
                    .pop_first()
                    .expect("nodes_with_min_edges must have at least one element"),
            );
            let mut component: HashSet<usize> = HashSet::new();
            while !component_candidates.is_empty() {
                let candidate = component_candidates
                    .pop_first()
                    .expect("component_candidates must have at least one element");
                component.insert(candidate);
                let valid_edge_nodes: BTreeSet<usize> = graph.edges[candidate]
                    .intersection(&HashSet::from_iter(nodes_with_min_edges.iter().copied()))
                    .copied()
                    .collect();
                component_candidates = &component_candidates | &valid_edge_nodes;
                nodes_with_min_edges = &nodes_with_min_edges - &valid_edge_nodes;
            }
            connected_components.push(component);
        }

        let mut clusters = connected_components;

        let unclustered_nodes: HashSet<usize> = &HashSet::from_iter(0..num_nodes)
            - &HashSet::from_iter(clusters.iter().flatten().copied());

        for node in unclustered_nodes {
            let node_in_clusters: Vec<_> = clusters
                .iter()
                .map(|cluster| {
                    graph.edges[node]
                        .iter()
                        .any(|edge_node| cluster.contains(edge_node))
                })
                .collect();

            if node_in_clusters.iter().any(|&x| x) {
                let i_cluster = *(0..clusters.len())
                    .collect::<Vec<usize>>()
                    .choose(&mut rng)
                    .unwrap();
                clusters[i_cluster].insert(node);
            }
        }

        let mut labels: Array1<i32> = -Array1::ones((num_nodes,));

        for (label, cluster) in clusters.iter().enumerate() {
            for i_cluster in cluster {
                labels[*i_cluster] = label as i32;
            }
        }

        labels
    }
}

#[pymethods]
impl SNG_DBSCAN {
    #[new]
    fn new(sampling_rate: f64, max_dist: f64, min_points: usize) -> Self {
        SNG_DBSCAN {
            sampling_rate,
            max_dist,
            min_points,
        }
    }

    #[pyo3(name = "fit_predict")]
    fn fit_predict_py<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
    ) -> Bound<'py, PyArray1<i32>> {
        self.fit_predict(x.as_array()).into_pyarray(py)
    }
}

#[pymodule]
fn sng_dbscan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<SNG_DBSCAN>()?;
    Ok(())
}

struct Graph<'a> {
    nodes: ArrayViewD<'a, f64>,
    edges: Vec<HashSet<usize>>,
}

impl Graph<'_> {
    fn new(nodes: ArrayViewD<'_, f64>) -> Graph<'_> {
        Graph {
            edges: nodes.iter().map(|_| HashSet::new()).collect(),
            nodes,
        }
    }

    fn add_edge(&mut self, index_1: usize, index_2: usize) {
        self.edges[index_1].insert(index_2);
        self.edges[index_2].insert(index_1);
    }
}
