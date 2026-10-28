# Deep Learning Network Anomaly Prediction — Project Explanation

This repository is a full machine-learning pipeline for early warning and detection of network anomalies from telemetry. The central idea is:

- capture per-second network measurements from a simulated router and endpoints,
- turn those measurements into historical windows,
- predict which anomaly type will start in the next 10 seconds,
- train and compare several models,
- evaluate them with realistic false-alarm constraints,
- and visualize the results in a replay dashboard.

The project is not just a single model notebook. It is a full data-science and ML workflow that includes synthetic data generation, cleaning, feature engineering, sequence preparation, cross-validation, model training, evaluation, saved models, and an interactive demo.

## 1. Project goal

The project aims to answer this question at each second of network operation:

> "Given the previous 30 seconds of telemetry, what anomaly is likely to start within the next 10 seconds?"

This is a forecasting task, not just a detection task. The model does not wait until the anomaly is already visible; it tries to warn earlier.

The anomaly classes are:

- normal
- congestion
- high_latency
- packet_loss
- bandwidth_degradation
- queue_buildup
- traffic_burst

The project ultimately compares:

- classical tabular baselines (logistic regression, random forest), and
- sequence models (LSTM, GRU, TCN)

against a common evaluation standard based on false alarms per hour.

## 2. What the project is doing at a high level

The workflow is conceptually:

1. Generate network telemetry runs using a Mininet-based simulation.
2. Store raw run data plus metadata describing the scenario and the underlying anomaly schedule.
3. Clean and normalize the data.
4. Build only causal, backward-looking features from each second.
5. Label each second as the future anomaly that starts within the next 10 seconds.
6. Build 30-second windows from the telemetry history.
7. Split runs into cross-validation folds by scenario and run number.
8. Train models for each fold and save predictions.
9. Evaluate them under a controlled false-alarm budget.
10. Use a final saved model and a rule-based detector in a replay app.

This is a complete end-to-end ML pipeline for time-series forecasting on network telemetry.

## 3. Main directory structure

Root directories and files:

- [common/](./common) — shared utilities for loading folds and model training
- [preprocessing/](./preprocessing) — all 7 data-preparation stages
- [models/](./models) — model definitions and evaluation tools
- [demo/](./demo) — live replay UI and rule-based detector
- [mininet_sim/](./mininet_sim) — telemetry generation and network experiment logic
- [data/](./data) — raw and processed datasets
- [results/](./results) — evaluation reports, predictions, calibration outputs, plots
- [experiments/](./experiments) — exploratory analysis scripts
- [docs/](./docs) — generated documentation artifacts
- [models/saved/](./models/saved) — trained final saved models
- [run_preprocessing.sh](./run_preprocessing.sh) — convenience script to run stages 1–7

## 4. The raw data: where it comes from

The project uses synthetic network data generated in a Mininet environment.

The simulation is defined in [mininet_sim/run_experiment.py](./mininet_sim/run_experiment.py), which creates router traffic, injection schedules, latency variation, queue buildup, and packet loss. It simulates a bottlenecked path between hosts and a router, then measures the system every second.

Key concepts in the simulator:

- Bottleneck: a single 50 Mbit/s link is the central network bottleneck.
- Background load: variable traffic is generated over time.
- Anomaly scenarios: each run is a particular network disturbance, such as congestion, packet loss, latency rise, traffic burst, or queue buildup.
- Metadata: for each row, the simulator stores information like scenario, phase, episode_id, and the injected settings.
- Baseline latency: each run is paired with a baseline latency reference, so the model learns relative latency rather than absolute differences across machines.

There are many scenarios, including:

- normal
- congestion
- high_latency
- packet_loss
- bandwidth_degradation
- queue_buildup
- traffic_burst
- mixed (used for demo / unseen test runs)

The older collector used in [mininet_sim/collect_telemetry.py](./mininet_sim/collect_telemetry.py) is a simplified telemetry sampler, but the main project uses the more advanced experiment generator in [mininet_sim/run_experiment.py](./mininet_sim/run_experiment.py).

## 5. The data model: what each row looks like

After cleaning, the processed telemetry contains rows describing one second of router and host behavior. It includes both raw measurements and labels.

The cleaned dataset is written to:

- [data/processed/network_telemetry_clean.csv](./data/processed)

Important columns include:

- run_id — unique identifier for a run
- machine — source machine (e.g. vbox or utm)
- scenario — scenario name
- elapsed_sec — time since the beginning of the run
- baseline_latency_ms — the nominal expected latency for that run
- latency_ms — current measured RTT / latency
- qdisc_backlog_bytes — router queue backlog
- tbf_dropped_per_sec — router drop rate
- netem_dropped_per_sec — packet loss due to random loss injection
- ingress_rx_bytes_per_sec — offered load into the router
- tx_bytes_per_sec — delivered traffic
- phase — normal, anomaly, cooldown, etc.
- label — whether the current row is within an anomaly episode
- anomaly_type — which class is active

A crucial design choice: metadata columns such as phase, episode_id, benign_excursion, injected_* values are intentionally not used as model features. They are kept for analysis and evaluation, but the model is trained only on measured telemetry features.

## 6. The feature pipeline: stages 1–7

The data preparation pipeline is implemented in [preprocessing/](./preprocessing). Each script is a stage, and they are meant to be run in order.

### Stage 1: Data cleaning

File: [preprocessing/01_data_cleaning.py](./preprocessing/01_data_cleaning.py)

This stage does the initial validation and cleanup.

It:

- loads raw CSV files and their JSON sidecars,
- associates each run with a unique run_id,
- validates required columns,
- converts columns to the right types,
- drops the first row of each run (because the initial per-second rates are undefined by construction),
- removes duplicate timestamps,
- marks invalid or negative values,
- splits runs at recording gaps (actual_delta_t > 1.5 seconds),
- repairs certain labels for traffic bursts,
- produces a clean telemetry data file and a data quality report.

Important rule: only the first row is discarded; missing values in the middle of a run are kept and flagged instead of silently deleting information. This preserves the time series without inventing a gap.

### Stage 2: Feature engineering

File: [preprocessing/02_feature_engineering.py](./preprocessing/02_feature_engineering.py)

This stage builds model-friendly features from measured telemetry. It does not use future information.

Examples of engineered features:

- latency_rel = latency_ms - baseline_latency_ms
- latency_max_rel
- latency_spread
- ping_loss_pct
- offered_mbit
- delivered_mbit
- utilization
- load_ratio
- shortfall
- queue_bytes / queue_packets / queue_delay_ms
- drop_ratio / loss_ratio
- rolling means and std devs over the last 5 or 10 seconds
- backward diffs such as latency_rel_diff
- missing_any flags to signal when a value was filled in during cleaning

It also compresses heavy-tailed values using signed log transforms for features like latency and queue metrics. This improves learning for deep networks and helps keep large values from dominating the scale.

The final output is:

- [data/processed/network_telemetry_features.csv](./data/processed)
- [data/processed/feature_columns.json](./data/processed)

### Stage 3: Future-onset label creation

File: [preprocessing/03_future_labels.py](./preprocessing/03_future_labels.py)

This is the core label logic.

Instead of predicting whether the current second is anomalous, the model predicts:

> “Which anomaly class starts in the next 10 seconds?”

At each anchor second t where the network is currently normal, the script checks whether an onset occurs between t and t + 10 seconds. If yes, the target is that anomaly class; otherwise the target is normal.

Design choices:

- It ignores seconds already inside an anomaly.
- It ignores cooldown periods after an anomaly.
- It drops incomplete horizons crossing run boundaries or gaps.
- It drops in-episode negatives, where the network is already degraded and calling it normal would be misleading.

Key output fields include:

- target
- target_name
- future_onset_sec
- lead_time_sec
- event_id
- anchor_status

This turns a detection problem into a multiclass forecasting problem that aligns tightly with the project’s early-warning objective.

### Stage 4: Sequence generation

File: [preprocessing/04_sequence_generation.py](./preprocessing/04_sequence_generation.py)

This converts each labeled anchor into a sequence window.

For each valid anchor:

- take the previous 30 seconds of data (LOOKBACK = 30),
- build a 3D tensor shaped as (30, number_of_features),
- assign the future-onset target as the label for that window,
- save metadata for that window.

The output is:

- [data/processed/sequences/X.npy](./data/processed/sequences)
- [data/processed/sequences/y.npy](./data/processed/sequences)
- [data/processed/sequences/metadata.csv](./data/processed/sequences)
- [data/processed/sequences/sequence_config.json](./data/processed/sequences)

This is the exact input format expected by the models.

### Stage 5: Cross-validation split by run

File: [preprocessing/05_cv_folds.py](./preprocessing/05_cv_folds.py)

The dataset is split at the run level, not the window level.

Why this matters:

- windows from the same run are correlated,
- if random split happens, leakage can occur,
- splitting by run ensures the model is evaluated on unseen runs.

The script creates 5 folds, assigning runs to train / validation / test roles by class and run number. The objective is to ensure every run is tested exactly once and no run appears in two roles in the same fold.

Important outputs:

- [data/processed/folds/folds.json](./data/processed/folds)
- [data/processed/folds/fold_assignment.csv](./data/processed/folds)

External runs (for example, runs from another machine, such as UTM) are kept separate from the main cross-validation and treated as a cross-machine test set.

### Stage 6: Per-fold scaling

File: [preprocessing/06_feature_scaling.py](./preprocessing/06_feature_scaling.py)

This stage fits a scaler on the training data only.

For each fold:

- compute mean and standard deviation on the fold’s training windows,
- apply them to validation and test windows,
- do not let validation/test statistics leak into training.

It also makes a scaler called full trained on all main-machine runs for the final model and the external evaluation.

The scaler output is:

- [data/processed/model_ready/scalers.json](./data/processed/model_ready)

This is a strict train-only scaling setup, which is essential for valid evaluation.

### Stage 7: Model-ready validation

File: [preprocessing/07_model_ready_validation.py](./preprocessing/07_model_ready_validation.py)

This stage is a verification guardrail. It checks whether the dataset is truly in a valid state before any model training.

It validates:

- no run appears in more than one role in a fold,
- every main-machine run is tested once,
- external runs are not in the fold,
- all classes are present in training,
- scaled features are finite,
- targets are valid and positive anchors point to valid future onsets,
- metadata columns are not accidentally included as features,
- windows are contiguous and correctly span 30 seconds of the same segment.

If any check fails, the script exits early.

## 7. Shared training utilities

The shared logic for model training and prediction is implemented in [common/train_utils.py](./common/train_utils.py) and [common/fold_data.py](./common/fold_data.py).

### common/fold_data.py

This module loads each fold and external data with the correct scaler applied.

It provides:

- load_fold(k)
- load_external()
- load_full_main()
- n_folds()
- CLASS_NAMES

This is how every model sees the same cross-validation data with identical preprocessing and scaling rules.

### common/train_utils.py

This module defines the common patterns used by all models:

- reproducible random seeds,
- class-weight calculation for class imbalance,
- save_predictions() for writing per-fold prediction files,
- quick_scores() for a quick F1 check,
- run_cv() to train each fold and save predictions,
- run_final() to train on all main-machine runs and save the final model,
- keras_fit() for recurrent models.

This is the backbone of the project’s training framework.

## 8. Model families

The project compares multiple model families.

### Logistic regression baseline

File: [models/01_classical_baselines.py](./models/01_classical_baselines.py)

The windowed sequence input is summarized into three statistics per feature:

- last value,
- average value over the window,
- trend = last - first

This yields a fixed-length tabular vector for classification.

The baseline is useful because it provides an interpretable, strong non-sequential benchmark.

### Random forest baseline

Also in [models/01_classical_baselines.py](./models/01_classical_baselines.py)

Random forest is trained on the same summarized feature representation. It often captures nonlinear thresholds well and is useful as a strong classical comparator.

### LSTM

File: [models/02_lstm.py](./models/02_lstm.py)

The network uses:

- an LSTM layer with 64 units,
- dropout,
- a dense hidden layer,
- a softmax output layer over 7 classes.

This model can capture temporal ordering and long-range dependencies in the 30-second history window.

### GRU

File: [models/03_gru.py](./models/03_gru.py)

Similar to the LSTM, but using a GRU unit instead. It has fewer parameters and often trains slightly faster.

### TCN

File: [models/04_tcn.py](./models/04_tcn.py)

The Temporal Convolutional Network uses causal convolutions with increasing dilation rates (1, 2, 4, 8). That gives a receptive field large enough to cover the whole 30-second lookback window while staying causal.

The TCN is a strong alternative to recurrent models because it can model temporal patterns with convolution instead of recurrent state.

## 9. Evaluation methodology

The evaluation framework is implemented in [models/evaluate.py](./models/evaluate.py).

This file is a very important part of the project because it answers the question: "Which model is actually better for early warning?"

### Operating point logic

The project does not compare models at a single naive threshold like argmax probability. Instead, it uses a false-alarm budget.

- Predictions are converted into anomaly alarms based on 1 - P(normal).
- For each fold, a threshold is selected on validation data so that false alarms stay below a target (for example 5 false alarms per hour of normal traffic).
- The same threshold is then applied to the test set for that fold.

This makes the comparison fair across models.

### Metrics reported

The evaluation file computes:

- macro F1 (across classes)
- anomaly F1 (binary anomaly detection quality)
- event detection rate
- mean lead time before onset
- false alarms per hour
- per-class precision / recall / F1
- confusion matrices
- plots of event detection and lead time distributions

The evaluation output is saved under:

- [results/evaluation/](./results/evaluation)

Some produced artifacts include:

- report.txt
- summary.csv
- per_class.csv
- events.csv
- thresholds.csv
- confusion plots
- event detection plots
- lead time plots

## 10. The demo app: replaying runs live

The most user-facing part of the project is the Streamlit app in [demo/replay_app.py](./demo/replay_app.py).

This app:

- replays a recorded run second by second,
- shows the actual telemetry traces,
- overlays the ground truth anomaly periods,
- overlays the model’s early warning signal,
- runs a rule-based detector to infer what is happening now,
- compares both outputs against the ground truth,
- produces an explanatory dashboard for operator-style analysis.

### Why there are two systems in the demo

The demo distinguishes between:

- Prediction: the model says an anomaly is likely to start within the next 10 seconds.
- Detection: a rule-based detector identifies an anomaly already happening right now.

These are complementary tasks.

### Rule-based detector

The detector logic is in [demo/detector.py](./demo/detector.py).

It uses rules based on telemetry, for example:

- RTT rising above baseline + 50 ms
- packet loss higher than 5%
- drop ratio exceeding 2%
- queue delay exceeding 50 ms
- shortfall ratio above 10%

A detection becomes visible only after enough consistent evidence accumulates, and the most common type is chosen from the recent evidence. This avoids false class assignments at the first second of a new anomaly.

### Demo features

The replay app includes:

- status banners for normal / warning / anomaly / recovering,
- probability chart for anomaly-start predictions,
- RTT trace and queue trace,
- offered load vs delivered load,
- a topology view of the router and links,
- timing tables for detection and warning history,
- evaluation verdicts for when the model was correct or early or late.

The app is meant to show causality and operational reasoning rather than just raw scores.

## 11. Model outputs and saved artifacts

The repository stores multiple layers of outputs in [results/](./results), including:

- prediction CSVs per model and fold
- threshold choices for each fold
- event-level outputs
- confusion matrices
- evaluation summaries
- model reports
- demo run predictions

Saved trained models live in [models/saved/](./models/saved), and are used by the demo to score new recordings.

## 12. Typical workflow to reproduce the project

The intended workflow is:

1. Generate or place raw telemetry runs under the data folders.
2. Run the preprocessing stages:

```bash
./run_preprocessing.sh
```

or step-by-step:

```bash
python preprocessing/01_data_cleaning.py
python preprocessing/02_feature_engineering.py
python preprocessing/03_future_labels.py
python preprocessing/04_sequence_generation.py
python preprocessing/05_cv_folds.py
python preprocessing/06_feature_scaling.py
python preprocessing/07_model_ready_validation.py
```

3. Train models:

```bash
python models/01_classical_baselines.py
python models/02_lstm.py
python models/03_gru.py
python models/04_tcn.py
```

4. Evaluate models:

```bash
python models/evaluate.py
```

5. Launch the UI:

```bash
streamlit run demo/replay_app.py
```

## 13. Why this project is structured this way

This project is deliberately designed around a few core principles:

- No leakage: the scaler fits on training data only.
- Run-level splitting: the test set must be unseen runs, not random windows.
- Future-onset labels: the task is predictive, not merely classification of currently ongoing anomalies.
- False-alarm budgeting: early-warning systems are judged by how often they cry wolf, not only by accuracy.
- Causal design: all features are based on the past; nothing uses future values.
- Operational transparency: the demo connects the model output to real network symptoms and physics.

These are all essential for evaluating network anomaly forecasting in a realistic way.

## 14. What makes this project distinctive

This project combines several ideas that are often separated in other ML projects:

- a network simulator for generating realistic anomaly incidents,
- a custom data pipeline for time-series anomaly forecasting,
- a multi-class prediction problem for future anomaly types,
- strict run-level validation and fold logic,
- a false-alarm-aware evaluation standard,
- a rule-based detector used for explanation and operator-facing analysis,
- a replay dashboard that visualizes the actual decision process.

In other words, this is not just a model training repo; it is a full research and demo pipeline for operational network anomaly forecasting.

## 15. Summary

At a glance, this project does the following:

- simulates network behavior under different anomaly conditions,
- records telemetry second-by-second,
- converts the data into causal 30-second windows,
- predicts which anomaly class is about to start within the next 10 seconds,
- compares sequential and classical models under a fixed false-alarm budget,
- visualizes the timing and quality of the predictions in a live replay app.

The result is an end-to-end early-warning system for network anomalies that can be used both for research and operational explanation.

## 16. Useful files to inspect next

If you want to understand the project in order, start here:

1. [preprocessing/01_data_cleaning.py](./preprocessing/01_data_cleaning.py)
2. [preprocessing/03_future_labels.py](./preprocessing/03_future_labels.py)
3. [preprocessing/04_sequence_generation.py](./preprocessing/04_sequence_generation.py)
4. [common/train_utils.py](./common/train_utils.py)
5. [models/02_lstm.py](./models/02_lstm.py)
6. [models/evaluate.py](./models/evaluate.py)
7. [demo/replay_app.py](./demo/replay_app.py)
8. [demo/detector.py](./demo/detector.py)

These files together tell the complete story of the project.
