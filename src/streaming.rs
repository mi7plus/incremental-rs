use crate::error::IncrementalError;
use crate::{IncrementalSupervisedEstimator};
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use rand::seq::SliceRandom;

/// Configuration options for streaming fit operations.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub batch_size: usize,
    pub shuffle_buffer_capacity: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            batch_size: 256,
            shuffle_buffer_capacity: 2048,
        }
    }
}

/// Convert a Polars DataFrame into ndarray structures.
fn df_to_ndarray_supervised(
    df: &DataFrame,
    feature_cols: &[&str],
    target_col: &str,
) -> Result<(Array2<f64>, Array1<f64>), IncrementalError> {
    let n_rows = df.height();
    let n_features = feature_cols.len();

    let mut x_vec = Vec::with_capacity(n_rows * n_features);
    for col_name in feature_cols {
        let series = df
            .column(col_name)
            .map_err(|_| IncrementalError::EmptyBatch)?
            .cast(&DataType::Float64)
            .map_err(|_| IncrementalError::EmptyBatch)?;
        let ca = series.f64().map_err(|_| IncrementalError::EmptyBatch)?;

        for val in ca.into_iter() {
            let v = val.ok_or(IncrementalError::NonFiniteInput)?;
            x_vec.push(v);
        }
    }

    // Convert column-major series reads into row-major matrix
    let mut x_array = Array2::zeros((n_rows, n_features));
    for (f_idx, _) in feature_cols.iter().enumerate() {
        for r_idx in 0..n_rows {
            x_array[[r_idx, f_idx]] = x_vec[f_idx * n_rows + r_idx];
        }
    }

    let target_series = df
        .column(target_col)
        .map_err(|_| IncrementalError::EmptyBatch)?
        .cast(&DataType::Float64)
        .map_err(|_| IncrementalError::EmptyBatch)?;
    let target_ca = target_series
        .f64()
        .map_err(|_| IncrementalError::EmptyBatch)?;

    let mut y_vec = Vec::with_capacity(n_rows);
    for val in target_ca.into_iter() {
        let v = val.ok_or(IncrementalError::NonFiniteInput)?;
        y_vec.push(v);
    }

    Ok((x_array, Array1::from(y_vec)))
}

/// Reservoir shuffle buffer designed to break sequence ordering bias during streaming updates.
pub struct StreamingShuffleBuffer {
    capacity: usize,
    buffer_x: Vec<Vec<f64>>,
    buffer_y: Vec<f64>,
}

impl StreamingShuffleBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer_x: Vec::with_capacity(capacity),
            buffer_y: Vec::with_capacity(capacity),
        }
    }

    pub fn push_batch(&mut self, batch_x: &Array2<f64>, batch_y: &Array1<f64>) {
        for (row, &y) in batch_x.rows().into_iter().zip(batch_y.iter()) {
            self.buffer_x.push(row.to_vec());
            self.buffer_y.push(y);
        }
    }

    /// Shuffles current buffer contents and extracts up to `batch_size` items.
    pub fn pop_batch(
        &mut self,
        batch_size: usize,
    ) -> Option<(Array2<f64>, Array1<f64>)> {
        if self.buffer_x.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..self.buffer_x.len()).collect();
        indices.shuffle(&mut rng);

        let take_count = batch_size.min(self.buffer_x.len());
        let drain_indices: Vec<usize> = indices.into_iter().take(take_count).collect();

        let n_features = self.buffer_x[0].len();
        let mut batch_x_mat = Array2::zeros((take_count, n_features));
        let mut batch_y_vec = Vec::with_capacity(take_count);

        for (out_idx, &src_idx) in drain_indices.iter().enumerate() {
            let row = &self.buffer_x[src_idx];
            for (f_idx, &val) in row.iter().enumerate() {
                batch_x_mat[[out_idx, f_idx]] = val;
            }
            batch_y_vec.push(self.buffer_y[src_idx]);
        }

        // Remove taken items (sorted descending to maintain correct indices)
        let mut sorted_drain = drain_indices;
        sorted_drain.sort_unstable_by(|a, b| b.cmp(a));
        for idx in sorted_drain {
            self.buffer_x.swap_remove(idx);
            self.buffer_y.swap_remove(idx);
        }

        Some((batch_x_mat, Array1::from(batch_y_vec)))
    }

    pub fn is_full(&self) -> bool {
        self.buffer_x.len() >= self.capacity
    }

    pub fn len(&self) -> usize {
        self.buffer_x.len()
    }
}

/// Execute streaming partial_fit directly over a LazyFrame.
pub fn fit_streaming_supervised<E: IncrementalSupervisedEstimator>(
    estimator: &mut E,
    lazy_frame: LazyFrame,
    feature_cols: &[&str],
    target_col: &str,
    config: StreamingConfig,
) -> Result<(), IncrementalError> {
    let df = lazy_frame
        .collect()
        .map_err(|_| IncrementalError::EmptyBatch)?;

    let (x_mat, y_vec) = df_to_ndarray_supervised(&df, feature_cols, target_col)?;
    let mut shuffle_buffer = StreamingShuffleBuffer::new(config.shuffle_buffer_capacity);

    // Process dataset through shuffle buffer
    for i in 0..x_mat.nrows() {
        let sample_x = x_mat.row(i).to_owned().insert_axis(Axis(0));
        let sample_y = Array1::from(vec![y_vec[i]]);
        shuffle_buffer.push_batch(&sample_x, &sample_y);

        if shuffle_buffer.is_full() {
            if let Some((b_x, b_y)) = shuffle_buffer.pop_batch(config.batch_size) {
                estimator.partial_fit(&b_x, &b_y)?;
            }
        }
    }

    // Flush remaining buffer items
    while shuffle_buffer.len() > 0 {
        if let Some((b_x, b_y)) = shuffle_buffer.pop_batch(config.batch_size) {
            estimator.partial_fit(&b_x, &b_y)?;
        }
    }

    Ok(())
}