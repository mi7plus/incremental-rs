#[derive(Debug, Clone)]
pub enum LearningRateSchedule {
    Constant {
        initial_rate: f64,
    },
    InverseScaling {
        initial_rate: f64,
        decay: f64,
        power: f64,
    },
}

impl LearningRateSchedule {
    pub fn calculate(&self, step: usize) -> f64 {
        match *self {
            LearningRateSchedule::Constant { initial_rate } => initial_rate,
            LearningRateSchedule::InverseScaling {
                initial_rate,
                decay,
                power,
            } => initial_rate / (1.0 + decay * (step as f64)).powf(power),
        }
    }
}