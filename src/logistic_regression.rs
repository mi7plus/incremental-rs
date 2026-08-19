use crate::error::IncrementalError;
use crate::learning_rate::LearningRateSchedule;
use crate::IncrementalSupervisedEstimator;
use ndarray::{Array1, Array2};
use std::collections::BTreeSet;

/// Multiclass strategy for incremental logistic regression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MulticlassStrategy {
    /// One-vs-Rest strategy: maintains an internal array of binary models,
    /// one per unique class encountered in the stream.
    OneVsRest,
}

#[derive(Debug)]
pub struct IncrementalLogisticRegression {
    // For binary classification (1 model) or OvR components
    weights: Option<Array2<f64>>, // Shape: (n_classes, n_features)
    bias: Option<Array1<f64>>,    // Shape: (n_classes)
    schedule: LearningRateSchedule,
    step_count: usize,
    l2_penalty: f64,
    strategy: MulticlassStrategy,
    seen_classes: BTreeSet<usize>,
}

impl IncrementalLogisticRegression {
    pub fn new(
        schedule: LearningRateSchedule,
        l2_penalty: f64,
        strategy: MulticlassStrategy,
    ) -> Self {
        Self {
            weights: None,
            bias: None,
            schedule,
            step_count: 0,
            l2_penalty,
            strategy,
            seen_classes: BTreeSet::new(),
        }
    }

    /// Returns the multiclass strategy configured for this model.
    pub fn strategy(&self) -> &MulticlassStrategy {
        &self.strategy
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Extends weight matrices dynamically if a new class label appears mid-stream.
    fn ensure_class_capacity(&mut self, label: usize, n_features: usize) {
        if !self.seen_classes.contains(&label) {
            self.seen_classes.insert(label);
            let target_rows = label + 1;

            match (&mut self.weights, &mut self.bias) {
                (Some(w), Some(b)) => {
                    if w.nrows() < target_rows {
                        let mut new_w = Array2::zeros((target_rows, n_features));
                        new_w.slice_mut(ndarray::s![..w.nrows(), ..]).assign(w);
                        *w = new_w;

                        let mut new_b = Array1::zeros(target_rows);
                        new_b.slice_mut(ndarray::s![..b.len()]).assign(b);
                        *b = new_b;
                    }
                }
                _ => {
                    self.weights = Some(Array2::zeros((target_rows, n_features)));
                    self.bias = Some(Array1::zeros(target_rows));
                }
            }
        }
    }

    fn validate_batch(
        &self,
        batch_x: &Array2<f64>,
        batch_y: &Array1<usize>,
    ) -> Result<(), IncrementalError> {
        if batch_x.nrows() == 0 {
            return Err(IncrementalError::EmptyBatch);
        }
        if batch_x.nrows() != batch_y.len() {
            return Err(IncrementalError::TargetDimensionMismatch {
                target_len: batch_y.len(),
                feature_rows: batch_x.nrows(),
            });
        }
        if batch_x.iter().any(|v| !v.is_finite()) {
            return Err(IncrementalError::NonFiniteInput);
        }
        if let Some(ref w) = self.weights {
            if batch_x.ncols() != w.ncols() {
                return Err(IncrementalError::DimensionMismatch {
                    expected: w.ncols(),
                    actual: batch_x.ncols(),
                });
            }
        }
        Ok(())
    }

    /// Fits a mini-batch with integer target labels (multiclass/binary).
    pub fn partial_fit_labels(
        &mut self,
        batch_x: &Array2<f64>,
        batch_y: &Array1<usize>,
    ) -> Result<(), IncrementalError> {
        self.validate_batch(batch_x, batch_y)?;

        let n_samples = batch_x.nrows() as f64;
        let n_features = batch_x.ncols();

        // Register any newly encountered class labels dynamically
        for &y in batch_y {
            self.ensure_class_capacity(y, n_features);
        }

        let weights = self.weights.as_mut().unwrap();
        let bias = self.bias.as_mut().unwrap();
        let eta = self.schedule.calculate(self.step_count);

        // One-vs-Rest Mini-Batch Gradient Step
        for &class_idx in &self.seen_classes {
            // Binary target indicator for this class
            let y_binary: Array1<f64> = batch_y
                .mapv(|y| if y == class_idx { 1.0 } else { 0.0 });

            let w_c = weights.row(class_idx);
            let b_c = bias[class_idx];

            // Linear combinations & activation: z = X*w + b
            let z = batch_x.dot(&w_c) + b_c;
            let probs = z.mapv(Self::sigmoid);

            // Gradient calculation for cross-entropy loss
            let errors = &probs - &y_binary;
            let mut w_grad = batch_x.t().dot(&errors) / n_samples;

            if self.l2_penalty > 0.0 {
                w_grad += &(&w_c * self.l2_penalty);
            }
            let b_grad = errors.sum() / n_samples;

            // Update class parameters
            let mut w_c_mut = weights.row_mut(class_idx);
            w_c_mut -= &(w_grad * eta);
            bias[class_idx] -= b_grad * eta;
        }

        self.step_count += 1;
        Ok(())
    }

    /// Predict class probabilities for input features.
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>, IncrementalError> {
        if x.nrows() == 0 {
            return Err(IncrementalError::EmptyBatch);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(IncrementalError::NonFiniteInput);
        }

        let (weights, bias) = match (&self.weights, &self.bias) {
            (Some(w), Some(b)) => (w, b),
            _ => return Err(IncrementalError::EmptyBatch),
        };

        if x.ncols() != weights.ncols() {
            return Err(IncrementalError::DimensionMismatch {
                expected: weights.ncols(),
                actual: x.ncols(),
            });
        }

        let mut raw_logits = x.dot(&weights.t()); // Shape: (n_samples, n_classes)
        for (i, &b) in bias.iter().enumerate() {
            raw_logits.column_mut(i).mapv_inplace(|v| Self::sigmoid(v + b));
        }

        Ok(raw_logits)
    }

    /// Predict discrete class labels.
    pub fn predict_labels(&self, x: &Array2<f64>) -> Result<Array1<usize>, IncrementalError> {
        let probs = self.predict_proba(x)?;
        let mut preds = Vec::with_capacity(x.nrows());

        for row in probs.rows() {
            let mut max_idx = 0;
            let mut max_prob = f64::NEG_INFINITY;
            for (idx, &p) in row.iter().enumerate() {
                if p > max_prob {
                    max_prob = p;
                    max_idx = idx;
                }
            }
            preds.push(max_idx);
        }

        Ok(Array1::from(preds))
    }
}

impl IncrementalSupervisedEstimator for IncrementalLogisticRegression {
    /// Satisfies trait for continuous 0.0/1.0 targets in binary classification.
    fn partial_fit(
        &mut self,
        batch_x: &Array2<f64>,
        batch_y: &Array1<f64>,
    ) -> Result<(), IncrementalError> {
        let labels = batch_y.mapv(|val| val.round() as usize);
        self.partial_fit_labels(batch_x, &labels)
    }

    /// Predicts target values as continuous class probabilities (for binary class 1).
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, IncrementalError> {
        let probs = self.predict_proba(x)?;
        if probs.ncols() > 1 {
            Ok(probs.column(1).to_owned())
        } else {
            Ok(probs.column(0).to_owned())
        }
    }
}