use incremental_rs::{
    fit_streaming_supervised, IncrementalLinearRegression, LearningRateSchedule, StreamingConfig,
};
use polars::prelude::*;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Streaming Linear Regression Example ===");

    let path = "synthetic_stream_data.parquet";
    let mut df = df!(
        "x1" => &(0..1000).map(|i| i as f64 * 0.1).collect::<Vec<_>>(),
        "x2" => &(0..1000).map(|i| i as f64 * -0.2).collect::<Vec<_>>(),
        "y"  => &(0..1000).map(|i| 2.0 * (i as f64 * 0.1) - 3.0 * (i as f64 * -0.2) + 0.5).collect::<Vec<_>>()
    )?;

    let file = File::create(path)?;
    ParquetWriter::new(file).finish(&mut df)?;

    let lazy_frame = LazyFrame::scan_parquet(path, ScanArgsParquet::default())?;
    let schedule = LearningRateSchedule::Constant { initial_rate: 0.001 };
    let mut model = IncrementalLinearRegression::new(schedule, 0.0);

    let config = StreamingConfig {
        batch_size: 64,
        shuffle_buffer_capacity: 256,
    };

    fit_streaming_supervised(&mut model, lazy_frame, &["x1", "x2"], "y", config)?;

    println!("Model fitted cleanly from parquet stream.");

    std::fs::remove_file(path)?;
    Ok(())
}