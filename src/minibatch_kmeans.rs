use crate::error::IncrementalError;
use crate::IncrementalUnsupervisedEstimator;
use ndarray::{Array1, Array2};
use rand::Rng;

#[derive(Debug)]
pub struct MiniBatchKMeans {
    n_clusters: usize,
    centroids: Option<Array2<f64>>,
    counts: Array1<f64>, // Cumulative point counts assigned to each cluster
    n_features: Option<usize>,
}

impl MiniBatchKMeans {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            centroids: None,
            counts: Array1::zeros(n_clusters),
            n_features: None,
        }
    }

    /// Calculates Euclidean distance squared between a single row and a centroid row.
    fn sq_euclidean(a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum()
    }

    /// Initialize centroids randomly from the first mini-batch.
    fn init_centroids(&mut self, batch_x: &Array2<f64>) -> Result<(), IncrementalError> {
        let n_samples = batch_x.nrows();
        let n_features = batch_x.ncols();

        if n_samples < self.n_clusters {
            return Err(IncrementalError::EmptyBatch);
        }

        let mut rng = rand::thread_rng();
        let mut centroids = Array2::zeros((self.n_clusters, n_features));
        let mut chosen_indices = Vec::with_capacity(self.n_clusters);

        while chosen_indices.len() < self.n_clusters {
            let idx = rng.gen_range(0..n_samples);
            if !chosen_indices.contains(&idx) {
                chosen_indices.push(idx);
            }
        }

        for (c_idx, &s_idx) in chosen_indices.iter().enumerate() {
            centroids.row_mut(c_idx).assign(&batch_x.row(s_idx));
        }

        self.centroids = Some(centroids);
        self.n_features = Some(n_features);
        Ok(())
    }

    fn validate_batch(&self, batch_x: &Array2<f64>) -> Result<(), IncrementalError> {
        if batch_x.nrows() == 0 {
            return Err(IncrementalError::EmptyBatch);
        }
        if batch_x.iter().any(|v| !v.is_finite()) {
            return Err(IncrementalError::NonFiniteInput);
        }
        if let Some(n_f) = self.n_features {
            if batch_x.ncols() != n_f {
                return Err(IncrementalError::DimensionMismatch {
                    expected: n_f,
                    actual: batch_x.ncols(),
                });
            }
        }
        Ok(())
    }

    /// Access calculated cluster centroids.
    pub fn centroids(&self) -> Option<&Array2<f64>> {
        self.centroids.as_ref()
    }
}

impl IncrementalUnsupervisedEstimator for MiniBatchKMeans {
    fn partial_fit(&mut self, batch_x: &Array2<f64>) -> Result<(), IncrementalError> {
        self.validate_batch(batch_x)?;

        if self.centroids.is_none() {
            self.init_centroids(batch_x)?;
        }

        let centroids = self.centroids.as_mut().unwrap();

        // 1. Assign samples to closest centroids
        let mut assignments = Vec::with_capacity(batch_x.nrows());
        for row in batch_x.rows() {
            let mut best_cluster = 0;
            let mut min_dist = f64::INFINITY;

            for (c_idx, c_row) in centroids.rows().into_iter().enumerate() {
                let dist = Self::sq_euclidean(&row, &c_row);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = c_idx;
                }
            }
            assignments.push(best_cluster);
        }

        // 2. Update centroids using Sculley's per-cluster learning rate
        for (row_idx, &c_idx) in assignments.iter().enumerate() {
            self.counts[c_idx] += 1.0;
            let eta = 1.0 / self.counts[c_idx]; // Per-cluster step size[cite: 1]

            let x_i = batch_x.row(row_idx);
            let mut c_i = centroids.row_mut(c_idx);

            // c_i = (1 - eta) * c_i + eta * x_i
            c_i.zip_mut_with(&x_i, |c_val, &x_val| {
                *c_val = (1.0 - eta) * (*c_val) + eta * x_val;
            });
        }

        Ok(())
    }

    fn predict_labels(&self, x: &Array2<f64>) -> Result<Array1<usize>, IncrementalError> {
        self.validate_batch(x)?;

        let centroids = match &self.centroids {
            Some(c) => c,
            None => return Err(IncrementalError::EmptyBatch),
        };

        let mut predictions = Vec::with_capacity(x.nrows());
        for row in x.rows() {
            let mut best_cluster = 0;
            let mut min_dist = f64::INFINITY;

            for (c_idx, c_row) in centroids.rows().into_iter().enumerate() {
                let dist = Self::sq_euclidean(&row, &c_row);
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = c_idx;
                }
            }
            predictions.push(best_cluster);
        }

        Ok(Array1::from(predictions))
    }
}