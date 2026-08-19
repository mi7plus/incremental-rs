use crate::error::IncrementalError;
use crate::learning_rate::LearningRateSchedule;
use crate::IncrementalSupervisedEstimator;
use ndarray::{Array1, Array2};

#[derive(Debug)]
pub struct IncrementalLinearRegression {
    weights: Option<Array1<f64>>,
    bias: f64,
    schedule: LearningRateSchedule,
    step_count: usize,
    l2_penalty: f64,
}

impl IncrementalLinearRegression {
    pub fn new(schedule: LearningRateSchedule, l2_penalty: f64) -> Self {
        Self {
            weights: None,
            bias: 0.0,
            schedule,
            step_count: 0,
            l2_penalty,
        }
    }

    fn validate_batch(
        &self,
        batch_x: &Array2<f64>,
        batch_y: &Array1<f64>,
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
        if batch_x.iter().any(|v| !v.is_finite()) || batch_y.iter().any(|v| !v.is_finite()) {
            return Err(IncrementalError::NonFiniteInput);
        }
        if let Some(ref w) = self.weights {
            if batch_x.ncols() != w.len() {
                return Err(IncrementalError::DimensionMismatch {
                    expected: w.len(),
                    actual: batch_x.ncols(),
                });
            }
        }
        Ok(())
    }
}

impl IncrementalSupervisedEstimator for IncrementalLinearRegression {
    fn partial_fit(
        &mut self,
        batch_x: &Array2<f64>,
        batch_y: &Array1<f64>,
    ) -> Result<(), IncrementalError> {
        self.validate_batch(batch_x, batch_y)?;

        let n_samples = batch_x.nrows() as f64;
        let n_features = batch_x.ncols();

        // Lazy-initialize weights on the first batch[cite: 1]
        let weights = self
            .weights
            .get_or_insert_with(|| Array1::zeros(n_features));

        // Compute predictions: y_hat = X * w + b
        let predictions = batch_x.dot(weights) + self.bias;
        let errors = &predictions - batch_y;

        // Compute gradients
        let mut weight_grad = batch_x.t().dot(&errors) / n_samples;
        if self.l2_penalty > 0.0 {
            weight_grad += &(weights.mapv(|w| w * self.l2_penalty)); // Ridge Regularization[cite: 1]
        }
        let bias_grad = errors.sum() / n_samples;

        // Update parameters using current schedule
        let eta = self.schedule.calculate(self.step_count);
        *weights -= &(weight_grad * eta);
        self.bias -= bias_grad * eta;

        self.step_count += 1;
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, IncrementalError> {
        if x.nrows() == 0 {
            return Err(IncrementalError::EmptyBatch);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(IncrementalError::NonFiniteInput);
        }
        let weights = match &self.weights {
            Some(w) => w,
            None => return Ok(Array1::zeros(x.nrows())),
        };

        if x.ncols() != weights.len() {
            return Err(IncrementalError::DimensionMismatch {
                expected: weights.len(),
                actual: x.ncols(),
            });
        }

        Ok(x.dot(weights) + self.bias)
    }
}