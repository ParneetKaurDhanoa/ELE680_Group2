import pandas as pd

# Load ETTh1 dataset
df = pd.read_csv("ETT-small/ETTh1.csv")

# Display first five rows
print("First 5 rows:")
print(df.head())

# Dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns.tolist())

# Data types
print("\nData types:")
print(df.dtypes)

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check duplicate timestamps
print("\nDuplicate timestamps:")
print(df["date"].duplicated().sum())

# Check date range
print("\nStart date:", df["date"].min())
print("End date:", df["date"].max())

# Check time intervals
print("\nTime interval counts:")
print(df["date"].diff().value_counts().head())

# Basic statistics
print("\nStatistical summary:")
print(df.describe())

import matplotlib.pyplot as plt

# Plot oil temperature over time
plt.figure(figsize=(14, 5))

plt.plot(df["date"], df["OT"])

plt.title("ETTh1 - Oil Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Oil Temperature (OT)")
plt.grid(True)

plt.tight_layout()

# Save the graph
plt.savefig("etth1_oil_temperature.png", dpi=300)

plt.close()

print("\nGraph saved successfully!")

# -------------------------------------
# ETTh1 - Short-term pattern analysis
# -------------------------------------

# Plot the first 7 days (168 hours)
plt.figure(figsize=(14, 5))
plt.plot(df["date"].iloc[:168], df["OT"].iloc[:168])

plt.title("ETTh1 - First 7 Days of Oil Temperature")
plt.xlabel("Date")
plt.ylabel("Oil Temperature (OT)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("etth1_first_7_days.png", dpi=300)
plt.close()


# Plot the first 30 days (720 hours)
plt.figure(figsize=(14, 5))
plt.plot(df["date"].iloc[:720], df["OT"].iloc[:720])

plt.title("ETTh1 - First 30 Days of Oil Temperature")
plt.xlabel("Date")
plt.ylabel("Oil Temperature (OT)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("etth1_first_30_days.png", dpi=300)
plt.close()


# Calculate daily average temperature
daily_avg = df.set_index("date")["OT"].resample("D").mean()

plt.figure(figsize=(14, 5))
plt.plot(daily_avg.index, daily_avg.values)

plt.title("ETTh1 - Daily Average Oil Temperature")
plt.xlabel("Date")
plt.ylabel("Daily Average OT")
plt.grid(True)
plt.tight_layout()
plt.savefig("etth1_daily_average.png", dpi=300)
plt.close()

print("\nThree additional graphs saved successfully!")