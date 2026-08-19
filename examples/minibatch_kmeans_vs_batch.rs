use incremental_rs::{IncrementalUnsupervisedEstimator, MiniBatchKMeans};
use ndarray::Array2;

fn main() {
    println!("=== Mini-Batch K-Means Example ===");

    // Clusters centered around (0,0) and (10,10)
    let batch = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.1, 0.2, -0.1, 0.0, 0.0, -0.2, 10.1, 9.9, 9.8, 10.2, 10.0, 10.1,
        ],
    )
    .unwrap();

    let mut kmeans = MiniBatchKMeans::new(2);

    for _ in 0..20 {
        kmeans.partial_fit(&batch).unwrap();
    }

    if let Some(centroids) = kmeans.centroids() {
        println!("Final Centroids calculated incrementally:");
        println!("{:?}", centroids);
    }
}