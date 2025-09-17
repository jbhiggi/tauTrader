# Methodology

## Methodology Overview

The methodology is organized into modular sections. Click a link below to jump directly to the details:

| Section | Description |
|---------|-------------|
| [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering) | Compute technical indicators, handle timestamps, clip outliers, and save the processed dataset. |
| [Sequence Creation and Data Splitting](#sequence-creation-and-data-splitting) | Split data into train/val/test while preserving temporal order, then prepare inputs for modeling. |
| [Normalization Diagnostics & Outputs](#normalization-diagnostics--outputs) | Apply robust and min-max scaling without leakage, verify distributions, and save metadata. |
| [Sequence Creation](#sequence-creation) | Generate rolling window sequences and aligned future-return labels for supervised learning. |
| [Class Weights](#class-weights) | Address class imbalance with computed or manual weight overrides. |
| [Model Architecture](#model-architecture) | LSTM-based binary classifier with regularization and dropout for temporal feature extraction. |
| [Model Training & Evaluation](#model-training--evaluation) | Training loop with early stopping, class weights, timing logs, and validation/test evaluation. |
| [Diagnostics & Reporting](#diagnostics--reporting) | Multi-page PDF plots and JSON reports for reproducible model evaluation. |
| [K-Fold Cross Validation & Threshold Mapping](#k-fold-cross-validation--threshold-mapping) | Derive a linear relation between validation and test thresholds (see [docs/kfold_cv.md](docs/kfold_cv.md)). |
| [Methodology Summary](#methodology-summary) | End-to-end recap of the pipeline and rationale. |


## Data Preprocessing & Feature Engineering

The raw OHLC (Open, High, Low, Close) time series is enriched with a suite of technical indicators and engineered features that serve as inputs for the machine learning pipeline. The key steps are:

### 1. Loading the Data
The OHLC dataset is read into a pandas DataFrame with timestamps parsed as datetime objects. This provides the base structure for all downstream transformations.

### 2. Feature Computation
Several well-established technical indicators are computed and appended to the DataFrame:

- **Simple Moving Average (SMA):**  
  A rolling mean of closing prices over a fixed lookback window (e.g., 20 days), providing a smoothed trend signal.

- **Moving Average Convergence Divergence (MACD):**  
  Calculated using a fast (12-period) and slow (26-period) exponential moving average, along with a 9-period signal line. This captures momentum shifts in price dynamics.

- **Bollinger Bands:**  
  Constructed from the SMA and standard deviation of prices over the same window (20 days), with upper and lower bands placed at ±2 standard deviations. These bands quantify relative volatility.

- **Stochastic Oscillator:**  
  A momentum indicator derived from the high, low, and closing prices over a 14-period lookback, with smoothing via a 3-period moving average. This measures the relative position of the close within the recent trading range.

- **Daily Returns:**  
  Percentage change in closing price from one day to the next, added as a new column. This can serve as an alternative target variable (`y`) in place of raw closing prices.

- **Timestamp Processing:**  
  The `timestamp` column is expanded into additional calendar-based features, including day of the week, to capture periodic effects in price movements.

### 3. Outlier Clipping
To mitigate the impact of extreme values, all features are clipped to specified lower and upper percentiles. This ensures that the model is not overly influenced by rare, high-magnitude fluctuations.

### 4. Saving the Processed Dataset
The final feature-augmented and clipped dataset is saved as a CSV (`apple_data_features_clipped.csv`), which serves as the standardized input for model training and evaluation.

---

## Sequence Creation and Data Splitting

### 1. Train / Validation / Test Split
The dataset is divided into three partitions:

- **Training set (70%)** — used to fit the model.  
- **Validation set (15%)** — used for model selection, threshold tuning, and hyperparameter optimization.  
- **Test set (15%)** — held out entirely from training and validation, serving as the final unbiased evaluation.

The split preserves the temporal order of the time series, ensuring no leakage of future information into past training windows.

### 2. Normalization
Normalization is applied to make features comparable in scale while preventing data leakage. Two complementary scaling strategies are used:

- **Robust Scaling (for heavy-tailed features):**  
  The `volume` column is scaled with a `RobustScaler`, which centers data on the median and scales according to the interquartile range. This reduces the influence of extreme outliers common in trading volume data.

- **Min-Max Scaling (for other features):**  
  All remaining features are scaled into a fixed range (default: \[-1, 1\]) using `MinMaxScaler`. This preserves the shape of distributions while ensuring features share a common numerical domain.

Both scalers are **fit only on the training set**, then applied to the validation and test sets, ensuring leak-free transformations.

### 3. Metadata Tracking
The fitted scaling parameters (centers, scales, min/max values) are saved into a JSON metadata file (`norm_meta.json`). This makes the normalization step reproducible and ensures that the same scaling can be applied consistently to new or unseen data.

---

At the end of this stage, three normalized datasets are available (`df_train_scaled`, `df_val_scaled`, `df_test_scaled`), along with the metadata required for reproducibility.

---

## Normalization Diagnostics & Outputs

After splitting and scaling, we verify the transformations with distribution‐comparison plots and persist the normalized datasets.

### 1) Distribution Checks (QC)

For each feature in the training, validation, and test splits, we generate side-by-side distribution plots comparing **raw** vs **normalized** values:

- **Training:** `figures/feature_norm_comparison_train.pdf`  
- **Validation:** `figures/feature_norm_comparison_val.pdf`  
- **Test:** `figures/feature_norm_comparison_test.pdf`

These PDFs (built with `plot_normalization_qc(..., bins=50)`) provide a quick visual audit that:
- The **heavy-tailed feature** (e.g., `volume`) is **median-centered** and scaled by IQR (RobustScaler), reducing outlier impact.
- **All other features** are mapped into the **specified MinMax range** (default \[-1, 1\]) without shape distortions beyond affine scaling.
- **Leakage control:** scalers fitted on **train only** produce consistent distributions on **val/test** without evidence of “peeking” (e.g., no unnatural compression/expansion unique to val/test).
- **Sanity cues:** flat lines or spikes suggest constant or near-constant features; extreme clipping suggests mis-specified ranges or upstream outliers.

**What to look for (pass/fail cues):**
- *PASS:* Train/val/test histograms show similar shapes post-scaling; ranges match the expected target scale; `volume` no longer dominated by a long right tail.  
- *FAIL:* Val/test ranges exceed train’s fitted Min/Max; dramatic distribution mismatches across splits; obvious clipping at bounds.

### 2) Persisting the Normalized Datasets

We save the normalized splits for downstream sequence building and model training:

- `model_data/apple_data_features_clipped_normalized_train.csv`  
- `model_data/apple_data_features_clipped_normalized_val.csv`  
- `model_data/apple_data_features_clipped_normalized_test.csv`

Coupled with the saved normalization metadata (`norm_meta.json`), these files guarantee **reproducibility** and **consistent scaling** when re-training or scoring on new data.

---

## Sequence Creation

We convert the normalized, time-ordered DataFrames into fixed-length rolling sequences and aligned future-move labels for supervised learning.

### Inputs
- **Features:** `config['data']['feature_cols']` (e.g., indicators, price/volume).
- **Window length:** `config['data']['window_size']` (number of timesteps per sample).
- **Prediction horizon:** `config['data']['lookahead']` (steps ahead to score the move).
- **Move threshold:** `config['data']['threshold']` (fractional return boundary).
- **Label direction:** `config['data']['direction']` (`"UP"` or `"DOWN"`).

### Labeling rule (binary)

Forward return is computed on the `close` column over the lookahead horizon. Let
$$
r_{t \to t+L}=\frac{\text{close}_{t+L}}{\text{close}_{t}}-1.
$$

r₍ₜ→ₜ₊ᴸ₎ = (close₍ₜ₊ᴸ₎ ÷ close₍ₜ₎) − 1

- If `direction="UP"`:  
  **label = 1** when r₍ₜ→ₜ₊ᴸ₎ ≥ threshold; otherwise **0**.  
- If `direction="DOWN"`:  
  **label = 1** when r₍ₜ→ₜ₊ᴸ₎ ≤ −threshold; otherwise **0**.  

> Note: the last `lookahead` rows cannot be labeled (future unknown) and are dropped before sequence rolling.

### Rolling window construction
From the trimmed feature matrix, we create overlapping windows using a sliding view:
- **Shape:** X ∈ ℝ⁽ᴺˢᵉᵠ × ʷⁱⁿᵈᵒʷˢⁱᶻᵉ × ᴺᶠᵉᵃᵗ⁾  
- The label y for each window is taken at the **last** timestep in that window (i.e., aligned to the window’s right edge).

### Outputs
For each split (train/val/test) we obtain:
- `X_*`: sequences of shape `(n_sequences, window_size, n_features)`.
- `y_*`: binary vector of shape `(n_sequences,)`.

### Leakage control & alignment
- Sequences are created **after** normalization with scalers fit on **train only**.
- Labels are computed from `close` using only **future** relative to each window’s start; windows whose target would peek beyond the dataset end are excluded.

### Quick diagnostics
When `verbose=True`, the function prints:
- Number of sequences and per-sequence shape.
- Class balance (via `np.bincount` on `y`).
- NaN/Inf checks on both `X` and `y`.

This step yields model-ready tensors for sequence models (e.g., 1D-CNNs, LSTMs, Transformers) while keeping the target definition explicit via `direction`, `threshold`, and `lookahead`.

---

## Class Weights

To address class imbalance, weights are computed from the training labels.  
- By default, class weights are set inversely proportional to class frequency.  
- A manual override can be provided via `config['training']['class_weight_override']`.  

These weights are later passed to the model’s loss function to balance learning across classes.

---

## Model Architecture

We employ a compact **LSTM-based classifier** tailored for **binary classification** of multivariate time-series windows:

- **Input:** sequences of length `window_size` with `n_features` per timestep.  
- **Core layer:** a single LSTM with configurable hidden units and L2 weight decay to extract temporal patterns.  
- **Dense + Dropout:** a 32-unit ReLU dense layer followed by dropout (`dropout_rate`) for regularization.  
- **Output:** a single sigmoid neuron producing probabilities for the positive class (`1` vs. `0`).  
- **Optimizer & loss:** Adam optimizer with configurable learning rate, trained under binary cross-entropy.  

This design balances expressive sequence modeling (via the LSTM) with lightweight regularization, making it suitable for small-to-medium datasets where overfitting and class imbalance are concerns.

### Metrics and Evaluation

During training, overall **accuracy** is monitored for stability. After training, we compute per-class **precision** and **recall** to capture the nuances of class imbalance:

- **Precision:** when the model predicts a class, how often is it correct?  
- **Recall:** of all true instances of a class, how many are captured?  

These class-level diagnostics highlight over- and under-prediction tendencies that global accuracy can mask. To ensure fairness across classes, we rely on the **arg-max rule** (selecting the most probable class) rather than fixed probability thresholds, mirroring inference-time behavior. This avoids artificially low recall on minority classes and keeps evaluation aligned with deployment.

Together, this architecture and metric suite provide both a high-level and class-specific view of performance, supporting informed adjustments to loss weighting or sampling strategies if needed.

---

## Model Training & Evaluation

Training is managed by the `run_training` wrapper, which fits the model while logging progress and (optionally) runtime:

- **Training data:** `(X_train, y_train)`  
- **Validation data:** `(X_val, y_val)`  
- **Configurable hyperparameters:** epochs, batch size, early stopping patience, and class weights.  
- **Monitoring:** if `verbose=True`, total training time is printed using the `time` package.  

After training, model performance is assessed on both **validation** and **test** sets:  
- Probabilities (`y_prob`) and hard predictions (`y_pred`) are generated.  
- Precision, recall, and decision thresholds are computed to guide later threshold selection.  

This ensures consistent evaluation across splits while capturing timing and convergence behavior.

---

## Diagnostics & Reporting

### 1) Multi-page Diagnostics PDF
We generate a consolidated PDF of validation and test diagnostics:

- **Precision–Recall (PR) curves** with threshold markers for both splits.  
- **Score distributions** (positive vs. negative) to visualize separability.  
- **Confusion matrices** at the default and selected thresholds.  
- **Training curves** (loss/metrics vs. epoch) from the Keras `history`.

**Output:** `figures/diagnostics.pdf`

This pack provides a fast visual audit of convergence, threshold behavior, and generalization (val → test).

### 2) Training & Evaluation Report (JSON)
A structured summary is produced for downstream analysis and reproducibility:

- **Best epoch** (from validation loss).  
- **Base rate** (positive class prevalence).  
- **Classification reports** (precision/recall/F1/support) for **val** and **test**.  
- **Threshold diagnostics** on **val** and **test**, including PR arrays and selected operating points.  
- **Target operating points:** evaluates performance at predefined recalls (e.g., 0.30 and 0.60).

**Output:** `model_data/report.json`

This report captures the key numeric results in a machine-readable format, enabling later steps such as **threshold selection** and **k-fold CV mapping** without re-running training.

---

## K-Fold Cross Validation & Threshold Mapping

After training and evaluation on the standard train/validation/test splits, we extend the analysis with **K-Fold Cross Validation (CV)**.

- Each fold yields its own optimal decision threshold (τ) on the validation set and the corresponding optimal τ on the test set.  
- Collecting results across folds allows us to fit a **linear mapping** between validation τ and test τ.  

In deployment, this mapping is applied as follows:  
1. Select the optimal τ\_val from the validation set.  
2. Apply the linear mapping to estimate the appropriate τ\_test.  
3. Use this mapped threshold as the operating point for live, unseen data.  

For a complete explanation of the **motivation, procedure, and visualizations**, see [kfold_cv.md](kfold_cv.md).  

---

## Methodology Summary

This methodology establishes a complete pipeline for transforming raw price data into reliable model outputs.  
- Data is preprocessed with technical features, outlier clipping, and normalization designed to prevent leakage.  
- Supervised sequences are created with aligned future-return labels.  
- A regularized LSTM classifier is trained with class weighting and early stopping to balance performance and generalization.  
- Thresholds are selected via validation, with a k-fold derived linear mapping applied to estimate robust test-time operating points.

Together, these steps ensure that the model is both **statistically sound** and **deployment-ready**, with reproducibility built in through saved configurations, scaling metadata, and diagnostic reports.
