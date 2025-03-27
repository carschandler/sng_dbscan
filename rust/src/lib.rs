use std::collections::HashSet;

use numpy::nalgebra::Norm;
use numpy::ndarray::{Array1, ArrayD, ArrayViewD, Axis};
use numpy::pyo3::Python;
use numpy::PyArray1;
use pyo3::prelude::*;
use rand::prelude::*;

#[pyclass(get_all, set_all)]
struct SNG_DBSCAN {
    sampling_rate: f64,
    max_dist: f64,
    min_points: u32,
}

impl SNG_DBSCAN {
    fn fit_predict(&self, x: ArrayViewD<f64>) -> Array1<i32> {
        let graph = Graph::new(x);
        let num_nodes = graph.nodes.len() as u32;
        let indices: Vec<u32> = (0..num_nodes).collect();
        let n_sample = self.sampling_rate.ceil() as usize;

        let mut rng = rand::rng();

        for (i_node, node) in graph.nodes.outer_iter().enumerate() {
            let i_sample = indices.iter().choose_multiple(&mut rng, n_sample);
            let sample = graph.nodes.select(Axis(0), &i_sample);
        }

        numpy::ndarray::array![1, 2]
    }
}

#[pymethods]
impl SNG_DBSCAN {
    #[new]
    fn new(sampling_rate: f64, max_dist: f64, min_points: u32) -> Self {
        SNG_DBSCAN {
            sampling_rate,
            max_dist,
            min_points,
        }
    }

    // #[pyfunction]
    // #[pyo3(name = "fit_predict")]
    // fn fit_predict_py<'py>() -> Bound<'py, PyArray1<f64>> {}
}

#[pymodule]
fn sng_dbscan(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<SNG_DBSCAN>()?;
    Ok(())
}

struct Graph<'a> {
    nodes: ArrayViewD<'a, f64>,
    edges: Vec<HashSet<i32>>,
}

impl Graph<'_> {
    fn new<'a>(nodes: ArrayViewD<'a, f64>) -> Graph<'a> {
        Graph {
            edges: nodes.iter().map(|_| HashSet::new()).collect(),
            nodes,
        }
    }
}
