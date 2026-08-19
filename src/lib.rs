pub mod error;
pub mod learning_rate;
pub mod linear_regression;
pub mod logistic_regression;
pub mod minibatch_kmeans;
pub mod monitoring;
pub mod naive_bayes;

#[cfg(feature = "polars-streaming")]
pub mod streaming;

pub use error::IncrementalError;
pub use learning_rate::LearningRateSchedule;
pub use linear_regression::IncrementalLinearRegression;
pub use logistic_regression::{IncrementalLogisticRegression, MulticlassStrategy};
pub use minibatch_kmeans::MiniBatchKMeans;
pub use monitoring::{BatchStats, MonitoredEstimator};
pub use naive_bayes::IncrementalGaussianNaiveBayes;

#[cfg(feature = "polars-streaming")]
pub use streaming::{fit_streaming_supervised, StreamingConfig, StreamingShuffleBuffer};

use ndarray::{Array1, Array2};

pub trait IncrementalSupervisedEstimator {
    fn partial_fit(
        &mut self,
        batch_x: &Array2<f64>,
        batch_y: &Array1<f64>,
    ) -> Result<(), IncrementalError>;

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, IncrementalError>;
}

pub trait IncrementalUnsupervisedEstimator {
    fn partial_fit(&mut self, batch_x: &Array2<f64>) -> Result<(), IncrementalError>;
    fn predict_labels(&self, x: &Array2<f64>) -> Result<Array1<usize>, IncrementalError>;
}