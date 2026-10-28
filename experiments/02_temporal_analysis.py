import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PHASE 1 - TEMPORAL ANOMALY ANALYSIS
# Project: Deep Learning-Based Early Prediction of
# Network Performance Anomalies in Data Center Networks
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

DATA_FILE = Path("data/processed/network_traffic_clean.csv")

FIGURE_DIR = Path("results/figures")
REPORT_FILE = Path("results/temporal_analysis_report.txt")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TEMPORAL ANOMALY ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values("time").reset_index(drop=True)

print("\nDataset loaded.")
print(f"Rows: {len(df)}")
print(f"Start: {df['time'].min()}")
print(f"End  : {df['time'].max()}")


# ------------------------------------------------------------
# 3. Basic temporal characteristics
# ------------------------------------------------------------

print("\n[1] TIME INTERVAL ANALYSIS")
print("-" * 70)

time_diff = df["time"].diff().dt.total_seconds().dropna()

print(f"Minimum interval : {time_diff.min()} seconds")
print(f"Median interval  : {time_diff.median()} seconds")
print(f"Maximum interval : {time_diff.max()} seconds")

print("\nInterval distribution:")
print(time_diff.value_counts().sort_index())


# ------------------------------------------------------------
# 4. Identify consecutive anomaly runs
# ------------------------------------------------------------

print("\n[2] ANOMALY RUN ANALYSIS")
print("-" * 70)

labels = df["label"].to_numpy()

runs = []

i = 0

while i < len(labels):

    if labels[i] == 1:

        start_index = i

        while i < len(labels) and labels[i] == 1:
            i += 1

        end_index = i - 1

        duration_seconds = (
            df.loc[end_index, "time"]
            - df.loc[start_index, "time"]
        ).total_seconds()

        run_length = end_index - start_index + 1

        runs.append({
            "start": df.loc[start_index, "time"],
            "end": df.loc[end_index, "time"],
            "length": run_length,
            "duration_seconds": duration_seconds
        })

    else:
        i += 1


runs_df = pd.DataFrame(runs)


if len(runs_df) > 0:

    print(f"Number of anomaly runs: {len(runs_df)}")

    print(
        f"Average run length: "
        f"{runs_df['length'].mean():.2f} observations"
    )

    print(
        f"Median run length: "
        f"{runs_df['length'].median():.2f} observations"
    )

    print(
        f"Maximum run length: "
        f"{runs_df['length'].max()} observations"
    )

    print(
        f"Average run duration: "
        f"{runs_df['duration_seconds'].mean():.2f} seconds"
    )

    print(
        f"Maximum run duration: "
        f"{runs_df['duration_seconds'].max():.2f} seconds"
    )

else:

    print("No anomaly runs detected.")


# ------------------------------------------------------------
# 5. Consecutive anomaly transition analysis
# ------------------------------------------------------------

print("\n[3] ANOMALY TRANSITION ANALYSIS")
print("-" * 70)

previous_label = df["label"].shift(1)

previous_anomaly = previous_label == 1

current_anomaly = df["label"] == 1

anomaly_after_anomaly = (
    previous_anomaly & current_anomaly
).sum()

total_previous_anomalies = previous_anomaly.sum()

if total_previous_anomalies > 0:

    transition_probability = (
        anomaly_after_anomaly
        / total_previous_anomalies
        * 100
    )

else:

    transition_probability = 0


print(
    f"Anomalous observations preceded by anomaly: "
    f"{anomaly_after_anomaly}"
)

print(
    f"P(current anomaly | previous anomaly): "
    f"{transition_probability:.2f}%"
)


# ------------------------------------------------------------
# 6. Normal → anomaly transitions
# ------------------------------------------------------------

normal_to_anomaly = (
    (previous_label == 0) &
    (df["label"] == 1)
).sum()

anomaly_to_normal = (
    (previous_label == 1) &
    (df["label"] == 0)
).sum()

print(
    f"\nNormal → Anomaly transitions: "
    f"{normal_to_anomaly}"
)

print(
    f"Anomaly → Normal transitions: "
    f"{anomaly_to_normal}"
)


# ------------------------------------------------------------
# 7. Time between anomaly events
# ------------------------------------------------------------

print("\n[4] TIME BETWEEN ANOMALY EVENTS")
print("-" * 70)

anomaly_times = df.loc[
    df["label"] == 1,
    "time"
]

anomaly_intervals = (
    anomaly_times
    .diff()
    .dt.total_seconds()
    .dropna()
)

if len(anomaly_intervals) > 0:

    print(
        f"Minimum gap : "
        f"{anomaly_intervals.min()} seconds"
    )

    print(
        f"Median gap  : "
        f"{anomaly_intervals.median()} seconds"
    )

    print(
        f"Maximum gap : "
        f"{anomaly_intervals.max()} seconds"
    )


# ------------------------------------------------------------
# 8. Future anomaly probability
# ------------------------------------------------------------

print("\n[5] FUTURE ANOMALY PROBABILITY")
print("-" * 70)

# Since the data interval is approximately 10 seconds,
# evaluate several future steps.

horizons = [1, 3, 6, 12, 30]

future_probabilities = []

for horizon in horizons:

    future_label = df["label"].shift(-horizon)

    valid = future_label.notna()

    probability = (
        future_label[valid].mean() * 100
    )

    future_probabilities.append({
        "horizon_steps": horizon,
        "approx_seconds": horizon * 10,
        "probability_percent": probability
    })

    print(
        f"+{horizon:2} steps "
        f"(~{horizon * 10:3} sec): "
        f"{probability:.2f}% anomaly"
    )


# ------------------------------------------------------------
# 9. Current anomaly → future anomaly persistence
# ------------------------------------------------------------

print("\n[6] ANOMALY PERSISTENCE")
print("-" * 70)

persistence_results = []

for horizon in horizons:

    future_label = df["label"].shift(-horizon)

    condition = df["label"] == 1

    valid = condition & future_label.notna()

    if valid.sum() > 0:

        probability = (
            future_label[valid].mean() * 100
        )

    else:

        probability = np.nan

    persistence_results.append({
        "horizon_steps": horizon,
        "approx_seconds": horizon * 10,
        "probability": probability
    })

    print(
        f"Given anomaly now → anomaly after "
        f"{horizon * 10} sec: "
        f"{probability:.2f}%"
    )


# ------------------------------------------------------------
# 10. Plot anomaly labels over time
# ------------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    df["time"],
    df["label"],
    linewidth=0.8
)

plt.title("Network Anomaly Labels Over Time")
plt.xlabel("Time")
plt.ylabel("Anomaly Label")

plt.yticks([0, 1], ["Normal", "Anomaly"])

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "temporal_anomaly_labels.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 11. Plot anomaly runs
# ------------------------------------------------------------

if len(runs_df) > 0:

    plt.figure(figsize=(10, 5))

    plt.hist(
        runs_df["length"],
        bins=30
    )

    plt.title("Distribution of Consecutive Anomaly Run Lengths")
    plt.xlabel("Number of Consecutive Anomalous Observations")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "anomaly_run_lengths.png",
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 12. Plot persistence
# ------------------------------------------------------------

persistence_df = pd.DataFrame(persistence_results)

plt.figure(figsize=(8, 5))

plt.plot(
    persistence_df["approx_seconds"],
    persistence_df["probability"],
    marker="o"
)

plt.title("Future Anomaly Probability Given Current Anomaly")
plt.xlabel("Future Horizon (seconds)")
plt.ylabel("Probability of Anomaly (%)")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "anomaly_persistence.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 13. Save report
# ------------------------------------------------------------

with open(REPORT_FILE, "w", encoding="utf-8") as report:

    report.write(
        "TEMPORAL ANOMALY ANALYSIS REPORT\n"
    )

    report.write("=" * 70 + "\n\n")

    report.write("DATASET\n")
    report.write("-" * 70 + "\n")
    report.write(f"Rows: {len(df)}\n")
    report.write(f"Start: {df['time'].min()}\n")
    report.write(f"End: {df['time'].max()}\n\n")

    report.write("TIME INTERVALS\n")
    report.write("-" * 70 + "\n")
    report.write(
        f"Minimum: {time_diff.min()} seconds\n"
    )
    report.write(
        f"Median: {time_diff.median()} seconds\n"
    )
    report.write(
        f"Maximum: {time_diff.max()} seconds\n\n"
    )

    report.write("ANOMALY RUNS\n")
    report.write("-" * 70 + "\n")

    if len(runs_df) > 0:

        report.write(
            f"Number of runs: {len(runs_df)}\n"
        )

        report.write(
            f"Average run length: "
            f"{runs_df['length'].mean():.2f}\n"
        )

        report.write(
            f"Median run length: "
            f"{runs_df['length'].median():.2f}\n"
        )

        report.write(
            f"Maximum run length: "
            f"{runs_df['length'].max()}\n"
        )

    report.write("\n")

    report.write("TRANSITIONS\n")
    report.write("-" * 70 + "\n")
    report.write(
        f"Normal → Anomaly: "
        f"{normal_to_anomaly}\n"
    )

    report.write(
        f"Anomaly → Normal: "
        f"{anomaly_to_normal}\n"
    )

    report.write(
        f"P(Anomaly | Previous Anomaly): "
        f"{transition_probability:.2f}%\n\n"
    )

    report.write("ANOMALY PERSISTENCE\n")
    report.write("-" * 70 + "\n")

    for result in persistence_results:

        report.write(
            f"+{result['approx_seconds']} sec: "
            f"{result['probability']:.2f}%\n"
        )


print("\n[7] OUTPUT FILES")
print("-" * 70)

print(f"Report: {REPORT_FILE}")

print("\nGenerated figures:")

for file in sorted(FIGURE_DIR.glob("*.png")):
    print(f"  {file.name}")


print("\n" + "=" * 70)
print("TEMPORAL ANALYSIS COMPLETED")
print("=" * 70)