use incremental_rs::{
    IncrementalError, IncrementalGaussianNaiveBayes, IncrementalLinearRegression,
    IncrementalLogisticRegression, IncrementalSupervisedEstimator, LearningRateSchedule,
    MulticlassStrategy,
};
use ndarray::{array, Array1, Array2};

#[test]
fn test_empty_batch_handling() {
    let empty_x: Array2<f64> = Array2::zeros((0, 2));
    let empty_y: Array1<f64> = Array1::zeros(0);

    let schedule = LearningRateSchedule::Constant { initial_rate: 0.01 };
    let mut linreg = IncrementalLinearRegression::new(schedule, 0.0);

    // Should gracefully fail with IncrementalError::EmptyBatch
    assert_eq!(
        linreg.partial_fit(&empty_x, &empty_y),
        Err(IncrementalError::EmptyBatch)
    );
    assert_eq!(
        linreg.predict(&empty_x),
        Err(IncrementalError::EmptyBatch)
    );
}

#[test]
fn test_non_finite_input_rejection() {
    let schedule = LearningRateSchedule::Constant { initial_rate: 0.01 };
    let mut linreg = IncrementalLinearRegression::new(schedule, 0.0);

    let nan_x = Array2::from_shape_vec((1, 2), vec![f64::NAN, 1.0]).unwrap();
    let valid_y = array![1.0];

    // Catch non-finite input explicitly without corrupting model state[cite: 1]
    assert_eq!(
        linreg.partial_fit(&nan_x, &valid_y),
        Err(IncrementalError::NonFiniteInput)
    );
}

#[test]
fn test_dimension_mismatch_across_batches() {
    let schedule = LearningRateSchedule::Constant { initial_rate: 0.01 };
    let mut logreg = IncrementalLogisticRegression::new(
        schedule,
        0.0,
        MulticlassStrategy::OneVsRest,
    );

    let batch1_x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let batch1_y = array![0, 1];

    logreg.partial_fit_labels(&batch1_x, &batch1_y).unwrap();

    // Mismatched feature dimensions in batch 2[cite: 1]
    let batch2_x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let batch2_y = array![0, 1];

    assert_eq!(
        logreg.partial_fit_labels(&batch2_x, &batch2_y),
        Err(IncrementalError::DimensionMismatch {
            expected: 2,
            actual: 3
        })
    );
}

#[test]
fn test_zero_variance_batch_naive_bayes() {
    let mut gnb = IncrementalGaussianNaiveBayes::new(1e-9);

    // All identical feature values (zero variance)[cite: 1]
    let zero_var_x = Array2::from_shape_vec((3, 2), vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0]).unwrap();
    let y = array![0, 0, 0];

    gnb.partial_fit(&zero_var_x, &y).unwrap();

    let test_x = Array2::from_shape_vec((1, 2), vec![5.0, 5.0]).unwrap();
    let preds = gnb.predict(&test_x).unwrap();

    // Should predict without numerical panics or NaN outputs[cite: 1]
    assert_eq!(preds, array![0]);
}

#[test]
fn test_single_sample_batch() {
    let schedule = LearningRateSchedule::Constant { initial_rate: 0.01 };
    let mut linreg = IncrementalLinearRegression::new(schedule, 0.0);

    let single_x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
    let single_y = array![5.0];

    // Single-sample mini-batch step should succeed cleanly[cite: 1]
    assert!(linreg.partial_fit(&single_x, &single_y).is_ok());
}