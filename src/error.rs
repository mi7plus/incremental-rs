use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum IncrementalError {
    #[error("Dimension mismatch: expected {expected} features, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty batch provided for incremental fitting")]
    EmptyBatch,

    #[error("Non-finite value (NaN or Infinity) detected in input data")]
    NonFiniteInput,

    #[error("Target batch dimension {target_len} does not match feature batch rows {feature_rows}")]
    TargetDimensionMismatch {
        target_len: usize,
        feature_rows: usize,
    },
}