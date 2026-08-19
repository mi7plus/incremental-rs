use crate::error::IncrementalError;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
struct ClassStats {
    count: usize,
    means: Array1<f64>,
    m2: Array1<f64>, // Sum of squared differences for Welford's algorithm[cite: 1]
}

impl ClassStats {
    fn new(n_features: usize) -> Self {
        Self {
            count: 0,
            means: Array1::zeros(n_features),
            m2: Array1::zeros(n_features),
        }
    }

    /// Parallel/Online Welford algorithm update[cite: 1]
    fn update_welford(&mut self, x: ArrayView2<f64>) {
        for row in x.rows() {
            self.count += 1;
            let delta = &row - &self.means;
            self.means += &(&delta / (self.count as f64));
            let delta2 = &row - &self.means;
            self.m2 += &(&delta * &delta2);
        }
    }

    fn variance(&self, var_smoothing: f64) -> Array1<f64> {
        if self.count < 2 {
            Array1::from_elem(self.means.len(), var_smoothing)
        } else {
            let sample_variance = &self.m2 / ((self.count - 1) as f64);
            // Apply variance smoothing floor to prevent divide-by-zero
            sample_variance.mapv(|v| (v + var_smoothing).max(var_smoothing))
        }
    }
}

#[derive(Debug)]
pub struct IncrementalGaussianNaiveBayes {
    classes: BTreeMap<usize, ClassStats>,
    var_smoothing: f64,
    n_features: Option<usize>,
}

impl IncrementalGaussianNaiveBayes {
    pub fn new(var_smoothing: f64) -> Self {
        Self {
            classes: BTreeMap::new(),
            var_smoothing,
            n_features: None,
        }
    }

    pub fn partial_fit(
        &mut self,
        batch_x: &Array2<f64>,
        batch_y: &Array1<usize>,
    ) -> Result<(), IncrementalError> {
        if batch_x.nrows() == 0 {
            return Err(IncrementalError::EmptyBatch);
        }
        if batch_x.nrows() != batch_y.len() {
            return Err(IncrementalError::TargetDimensionMismatch {
                target_len: batch_y.len(),
                feature_rows: batch_x.nrows(),
            });
        }
        if batch_x.iter().any(|v| !v.is_finite()) {
            return Err(IncrementalError::NonFiniteInput);
        }

        let num_features = *self.n_features.get_or_insert(batch_x.ncols());
        if batch_x.ncols() != num_features {
            return Err(IncrementalError::DimensionMismatch {
                expected: num_features,
                actual: batch_x.ncols(),
            });
        }

        // Group rows by class target and update running statistics[cite: 1]
        for (&label, row) in batch_y.iter().zip(batch_x.rows()) {
            let stats = self
                .classes
                .entry(label)
                .or_insert_with(|| ClassStats::new(num_features));
            let sample = row.insert_axis(Axis(0));
            stats.update_welford(sample);
        }

        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<usize>, IncrementalError> {
        if x.nrows() == 0 {
            return Err(IncrementalError::EmptyBatch);
        }
        let n_features = match self.n_features {
            Some(f) => f,
            None => return Err(IncrementalError::EmptyBatch),
        };

        if x.ncols() != n_features {
            return Err(IncrementalError::DimensionMismatch {
                expected: n_features,
                actual: x.ncols(),
            });
        }

        let total_samples: usize = self.classes.values().map(|s| s.count).sum();
        if total_samples == 0 {
            return Err(IncrementalError::EmptyBatch);
        }

        let mut predictions = Vec::with_capacity(x.nrows());

        for row in x.rows() {
            let mut best_class = 0;
            let mut best_log_prob = f64::NEG_INFINITY;

            for (&class_label, stats) in &self.classes {
                let prior_log = (stats.count as f64 / total_samples as f64).ln();
                let vars = stats.variance(self.var_smoothing);

                let mut log_likelihood = 0.0;
                for i in 0..n_features {
                    let diff = row[i] - stats.means[i];
                    let var = vars[i];
                    log_likelihood +=
                        -0.5 * (2.0 * std::f64::consts::PI * var).ln() - (diff * diff) / (2.0 * var);
                }

                let total_log_prob = prior_log + log_likelihood;
                if total_log_prob > best_log_prob {
                    best_log_prob = total_log_prob;
                    best_class = class_label;
                }
            }
            predictions.push(best_class);
        }

        Ok(Array1::from(predictions))
    }
}