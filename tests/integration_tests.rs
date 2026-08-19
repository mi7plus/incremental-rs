use incremental_rs::{
    IncrementalGaussianNaiveBayes, IncrementalLinearRegression, IncrementalLogisticRegression,
    IncrementalSupervisedEstimator, IncrementalUnsupervisedEstimator, LearningRateSchedule,
    MiniBatchKMeans, MonitoredEstimator, MulticlassStrategy,
};
use ndarray::{array, Array2};

#[test]
fn test_convergence_monitoring_callback() {
    let x = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
    let y = array![3.0, 7.0, 11.0, 15.0];

    let schedule = LearningRateSchedule::Constant { initial_rate: 0.01 };
    let mut model = IncrementalLinearRegression::new(schedule, 0.0);

    let mut logged_steps = Vec::new();

    {
        let mut monitored = MonitoredEstimator::new(&mut model, |stats| {
            logged_steps.push(stats.step);
        });

        for _ in 0..5 {
            monitored.partial_fit(&x, &y).unwrap();
        }
    }

    // Verify callback was triggered 5 times in sequential step increments[cite: 1]
    assert_eq!(logged_steps, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_minibatch_kmeans_cluster_separation() {
    // Two distinct clusters: near (0,0) and near (10,10)
    let b1 = Array2::from_shape_vec((4, 2), vec![0.0, 0.1, 0.1, 0.0, 10.0, 10.1, 9.9, 10.0]).unwrap();
    let b2 = Array2::from_shape_vec((4, 2), vec![0.2, -0.1, -0.1, 0.1, 10.2, 9.8, 10.1, 9.9]).unwrap();

    let mut kmeans = MiniBatchKMeans::new(2);

    // Fit across mini-batches
    for _ in 0..50 {
        kmeans.partial_fit(&b1).unwrap();
        kmeans.partial_fit(&b2).unwrap();
    }

    let test_points = Array2::from_shape_vec((2, 2), vec![0.05, 0.05, 10.05, 10.05]).unwrap();
    let labels = kmeans.predict_labels(&test_points).unwrap();

    // Verify samples land in separate clusters
    assert_ne!(labels[0], labels[1]);
}

#[test]
fn test_logistic_regression_binary_convergence() {
    // Cluster 0 centered around (-2, -2), Cluster 1 centered around (2, 2)
    let x_data = Array2::from_shape_vec(
        (6, 2),
        vec![-2.0, -1.8, -1.9, -2.1, -2.2, -1.9, 2.0, 1.8, 1.9, 2.1, 2.2, 1.9],
    )
    .unwrap();
    let y_data = array![0, 0, 0, 1, 1, 1];

    let schedule = LearningRateSchedule::Constant { initial_rate: 0.1 };
    let mut model = IncrementalLogisticRegression::new(
        schedule,
        0.001,
        MulticlassStrategy::OneVsRest,
    );

    // Fit mini-batches
    for _ in 0..300 {
        model.partial_fit_labels(&x_data, &y_data).unwrap();
    }

    let preds = model.predict_labels(&x_data).unwrap();
    assert_eq!(preds, y_data);
}

#[test]
fn test_incremental_linear_regression_convergence() {
    // Ground truth function: y = 2.0 * x1 - 3.0 * x2 + 0.5
    let x_data = Array2::from_shape_vec(
        (6, 2),
        vec![
            1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 5.0, 2.0, 4.0, 1.0, 6.0, 3.0,
        ],
    )
    .unwrap();
    let y_data = array![-3.5, 1.5, -5.5, 4.5, 5.5, 3.5];

    let schedule = LearningRateSchedule::Constant { initial_rate: 0.05 };
    let mut model = IncrementalLinearRegression::new(schedule, 0.0);

    // Fit incrementally in 3 mini-batches of size 2[cite: 1]
    for step in 0..1000 {
        let idx = (step * 2) % 6;
        let batch_x = x_data.slice(ndarray::s![idx..idx + 2, ..]).to_owned();
        let batch_y = y_data.slice(ndarray::s![idx..idx + 2]).to_owned();
        model.partial_fit(&batch_x, &batch_y).unwrap();
    }

    let preds = model.predict(&x_data).unwrap();
    for (y_true, y_pred) in y_data.iter().zip(preds.iter()) {
        assert!((y_true - y_pred).abs() < 0.2);
    }
}

#[test]
fn test_gaussian_naive_bayes_midstream_class() {
    let mut gnb = IncrementalGaussianNaiveBayes::new(1e-9);

    // Batch 1: Only class 0 observations[cite: 1]
    let b1_x = Array2::from_shape_vec((2, 2), vec![1.0, 1.1, 0.9, 0.8]).unwrap();
    let b1_y = array![0, 0];
    gnb.partial_fit(&b1_x, &b1_y).unwrap();

    // Batch 2: New class 1 introduced mid-stream[cite: 1]
    let b2_x = Array2::from_shape_vec((2, 2), vec![5.0, 5.2, 4.8, 5.1]).unwrap();
    let b2_y = array![1, 1];
    gnb.partial_fit(&b2_x, &b2_y).unwrap();

    let test_x = Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 5.0, 5.0]).unwrap();
    let preds = gnb.predict(&test_x).unwrap();

    assert_eq!(preds, array![0, 1]);
}

#[cfg(feature = "polars-streaming")]
#[test]
fn test_polars_streaming_integration() {
    use incremental_rs::{
        fit_streaming_supervised, IncrementalLinearRegression, LearningRateSchedule, StreamingConfig,
    };
    use polars::prelude::*;

    let df = df!(
        "x1" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x2" => &[2.0, 1.0, 4.0, 2.0, 1.0, 3.0],
        "y"  => &[-3.5, 1.5, -5.5, 4.5, 5.5, 3.5]
    )
    .unwrap();

    let lazy_frame = df.lazy();
    let schedule = LearningRateSchedule::Constant { initial_rate: 0.01 };
    let mut model = IncrementalLinearRegression::new(schedule, 0.0);

    let config = StreamingConfig {
        batch_size: 2,
        shuffle_buffer_capacity: 4,
    };

    fit_streaming_supervised(
        &mut model,
        lazy_frame,
        &["x1", "x2"],
        "y",
        config,
    )
    .unwrap();
}