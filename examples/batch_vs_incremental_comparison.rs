use incremental_rs::{
    BatchStats, IncrementalLinearRegression, IncrementalSupervisedEstimator,
    LearningRateSchedule, MonitoredEstimator,
};
use ndarray::{array, Array2};

fn main() {
    println!("===Batch vs. Incremental Convergence Comparison ===");

    let x = Array2::from_shape_vec(
        (8, 2),
        vec![
            1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 5.0, 2.0, 4.0, 1.0, 6.0, 3.0, 7.0, 2.0, 8.0, 5.0,
        ],
    )
    .unwrap();
    let y = array![-3.5, 1.5, -5.5, 4.5, 5.5, 3.5, 8.5, 1.5];

    let schedule = LearningRateSchedule::InverseScaling {
        initial_rate: 0.05,
        decay: 0.01,
        power: 0.5,
    };

    let mut model = IncrementalLinearRegression::new(schedule, 0.01);

    println!("Step | Mini-Batch MSE Loss");
    println!("--------------------------");

    let mut monitored = MonitoredEstimator::new(&mut model, |stats: BatchStats| {
        println!("{:4} | {:.6}", stats.step, stats.loss);
    });

    // Run multiple incremental epoch passes over data[cite: 1]
    for _ in 0..10 {
        monitored.partial_fit(&x, &y).unwrap();
    }
}