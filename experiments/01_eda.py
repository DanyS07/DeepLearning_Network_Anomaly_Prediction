import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PHASE 1 - EXPLORATORY DATA ANALYSIS
# Project: Deep Learning-Based Early Prediction of
# Network Performance Anomalies in Data Center Networks
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

DATA_FILE = Path("data/processed/network_traffic_clean.csv")

FIGURE_DIR = Path("results/figures")
REPORT_FILE = Path("results/eda_report.txt")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 1: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

df["time"] = pd.to_datetime(df["time"])

print("\nDataset loaded successfully.")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ------------------------------------------------------------
# 3. Feature groups
# ------------------------------------------------------------

continuous_features = [
    "duration",
    "packet_count",
    "bytes_sent",
    "bytes_received",
    "bytes_per_packet"
]

identifier_features = [
    "source_ip_int",
    "destination_ip_int"
]

network_discrete_features = [
    "source_port",
    "destination_port",
    "protocol"
]


# ------------------------------------------------------------
# 4. Basic statistics
# ------------------------------------------------------------

print("\n[1] BASIC STATISTICS")
print("-" * 70)

print(df[continuous_features].describe())


# ------------------------------------------------------------
# 5. Label distribution
# ------------------------------------------------------------

print("\n[2] LABEL DISTRIBUTION")
print("-" * 70)

label_counts = df["label"].value_counts().sort_index()

print(label_counts)

label_percentages = (
    df["label"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nPercentages:")

for label, percentage in label_percentages.items():
    print(f"Label {label}: {percentage:.2f}%")


# ------------------------------------------------------------
# 6. Feature statistics by label
# ------------------------------------------------------------

print("\n[3] FEATURE STATISTICS BY LABEL")
print("-" * 70)

grouped_statistics = df.groupby("label")[continuous_features].mean()

print(grouped_statistics)


# ------------------------------------------------------------
# 7. Label distribution plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

df["label"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Network Traffic Label Distribution")
plt.xlabel("Label")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "label_distribution.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 8. Feature distributions
# ------------------------------------------------------------

df[continuous_features].hist(
    figsize=(12, 8),
    bins=40
)

plt.suptitle("Distributions of Network Traffic Features")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "feature_distributions.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 9. Boxplots by label
# ------------------------------------------------------------

for feature in continuous_features:

    plt.figure(figsize=(7, 5))

    df.boxplot(
        column=feature,
        by="label"
    )

    plt.title(f"{feature} by Network Traffic Label")
    plt.suptitle("")
    plt.xlabel("Label")
    plt.ylabel(feature)

    plt.tight_layout()

    filename = f"{feature}_boxplot.png"

    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 10. Anomaly occurrence over time
# ------------------------------------------------------------

# Convert labels into a time series.
# We count anomalous observations within 5-minute intervals.

time_series = (
    df.set_index("time")
    .resample("5min")["label"]
    .agg(
        total_records="count",
        anomalies="sum"
    )
    .reset_index()
)

plt.figure(figsize=(12, 5))

plt.plot(
    time_series["time"],
    time_series["anomalies"]
)

plt.title("Anomalous Network Traffic Over Time")
plt.xlabel("Time")
plt.ylabel("Number of Anomalous Records")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "anomaly_over_time.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 11. Anomaly percentage over time
# ------------------------------------------------------------

time_series["anomaly_percentage"] = (
    time_series["anomalies"]
    / time_series["total_records"]
    * 100
)

plt.figure(figsize=(12, 5))

plt.plot(
    time_series["time"],
    time_series["anomaly_percentage"]
)

plt.title("Anomaly Percentage Over Time")
plt.xlabel("Time")
plt.ylabel("Anomaly Percentage (%)")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "anomaly_percentage_over_time.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 12. Correlation analysis
# ------------------------------------------------------------

correlation = df[continuous_features + ["label"]].corr()

print("\n[4] CORRELATION WITH LABEL")
print("-" * 70)

print(
    correlation["label"]
    .sort_values(ascending=False)
)


# ------------------------------------------------------------
# 13. Protocol distribution
# ------------------------------------------------------------

print("\n[5] PROTOCOL DISTRIBUTION")
print("-" * 70)

print(
    df["protocol"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 14. Port analysis
# ------------------------------------------------------------

print("\n[6] PORT ANALYSIS")
print("-" * 70)

print(
    "Unique source ports      :",
    df["source_port"].nunique()
)

print(
    "Unique destination ports :",
    df["destination_port"].nunique()
)


# ------------------------------------------------------------
# 15. Temporal statistics
# ------------------------------------------------------------

print("\n[7] TEMPORAL ANALYSIS")
print("-" * 70)

time_difference = df["time"].diff().dt.total_seconds()

print(
    "Minimum time difference (seconds):",
    time_difference.min()
)

print(
    "Median time difference (seconds):",
    time_difference.median()
)

print(
    "Maximum time difference (seconds):",
    time_difference.max()
)


# ------------------------------------------------------------
# 16. Save EDA report
# ------------------------------------------------------------

with open(REPORT_FILE, "w", encoding="utf-8") as report:

    report.write("PHASE 1 - EXPLORATORY DATA ANALYSIS REPORT\n")
    report.write("=" * 70 + "\n\n")

    report.write("DATASET OVERVIEW\n")
    report.write("-" * 70 + "\n")
    report.write(f"Rows: {df.shape[0]}\n")
    report.write(f"Columns: {df.shape[1]}\n")
    report.write(f"Start time: {df['time'].min()}\n")
    report.write(f"End time: {df['time'].max()}\n\n")

    report.write("LABEL DISTRIBUTION\n")
    report.write("-" * 70 + "\n")
    report.write(str(label_counts))
    report.write("\n\n")

    report.write("LABEL PERCENTAGES\n")
    report.write("-" * 70 + "\n")
    report.write(str(label_percentages))
    report.write("\n\n")

    report.write("MEAN FEATURES BY LABEL\n")
    report.write("-" * 70 + "\n")
    report.write(str(grouped_statistics))
    report.write("\n\n")

    report.write("CORRELATION WITH LABEL\n")
    report.write("-" * 70 + "\n")
    report.write(str(correlation["label"].sort_values(ascending=False)))
    report.write("\n\n")

    report.write("PROTOCOL DISTRIBUTION\n")
    report.write("-" * 70 + "\n")
    report.write(str(df["protocol"].value_counts().sort_index()))
    report.write("\n\n")

    report.write("PORT INFORMATION\n")
    report.write("-" * 70 + "\n")
    report.write(
        f"Unique source ports: {df['source_port'].nunique()}\n"
    )
    report.write(
        f"Unique destination ports: "
        f"{df['destination_port'].nunique()}\n"
    )
    report.write("\n")

    report.write("TEMPORAL INFORMATION\n")
    report.write("-" * 70 + "\n")
    report.write(
        f"Minimum time difference: "
        f"{time_difference.min()} seconds\n"
    )
    report.write(
        f"Median time difference: "
        f"{time_difference.median()} seconds\n"
    )
    report.write(
        f"Maximum time difference: "
        f"{time_difference.max()} seconds\n"
    )


# ------------------------------------------------------------
# 17. Completion
# ------------------------------------------------------------

print("\n[8] OUTPUT FILES")
print("-" * 70)

print(f"EDA report:")
print(REPORT_FILE)

print("\nFigures generated:")

for file in sorted(FIGURE_DIR.glob("*.png")):
    print(f"  {file.name}")

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)