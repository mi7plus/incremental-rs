use crate::error::IncrementalError;
use crate::IncrementalSupervisedEstimator;
use ndarray::{Array1, Array2};

/// Metrics recorded after each batch execution.
#[derive(Debug, Clone, Copy)]
pub struct BatchStats {
    pub step: usize,
    pub loss: f64,
    pub learning_rate: Option<f64>,
}

/// A wrapper that attaches a monitoring callback to any supervised incremental model.
pub struct MonitoredEstimator<'a, E: IncrementalSupervisedEstimator, F>
where
    F: FnMut(BatchStats),
{
    pub estimator: &'a mut E,
    pub callback: F,
    pub step_counter: usize,
}

impl<'a, E: IncrementalSupervisedEstimator, F> MonitoredEstimator<'a, E, F>
where
    F: FnMut(BatchStats),
{
    pub fn new(estimator: &'a mut E, callback: F) -> Self {
        Self {
            estimator,
            callback,
            step_counter: 0,
        }
    }

    /// Calculates Mean Squared Error loss between true and predicted targets.
    fn calculate_mse(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        if y_true.is_empty() {
            return 0.0;
        }
        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            / (y_true.len() as f64)
    }
}

impl<'a, E: IncrementalSupervisedEstimator, F> IncrementalSupervisedEstimator
    for MonitoredEstimator<'a, E, F>
where
    F: FnMut(BatchStats),
{
    fn partial_fit(
        &mut self,
        batch_x: &Array2<f64>,
        batch_y: &Array1<f64>,
    ) -> Result<(), IncrementalError> {
        // 1. Predict current batch before parameter update to evaluate pre-fit step loss
        let predictions = self.estimator.predict(batch_x).unwrap_or_default();
        let loss = Self::calculate_mse(batch_y, &predictions);

        // 2. Perform parameter update step
        self.estimator.partial_fit(batch_x, batch_y)?;

        // 3. Emit step metrics via callback
        (self.callback)(BatchStats {
            step: self.step_counter,
            loss,
            learning_rate: None,
        });

        self.step_counter += 1;
        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, IncrementalError> {
        self.estimator.predict(x)
    }
}