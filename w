import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("weather.csv")

# Fill missing values if any
df.fillna(df.mean(numeric_only=True), inplace=True)

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

# Grouping data by year
grouped = df.groupby("year").agg({
    "temperature": "mean",
    "rainfall": "sum",
    "humidity": "mean"
}).reset_index()

# Prepare for Linear Regression (prediction)
X = grouped[["year"]]
y = grouped["temperature"]
model = LinearRegression()
model.fit(X, y)
predicted = model.predict(X)

# Plotting in 2x2 layout
fig, axs = plt.subplots(2, 2, figsize=(14, 7))
fig.tight_layout(pad=4)

# 1. Temperature Line Chart
axs[0, 0].plot(grouped["year"], grouped["temperature"], color='red', marker='o', label='Temperature (°C)')
axs[0, 0].set_title("Temperature Trends Over Years")
axs[0, 0].set_xlabel("Year")
axs[0, 0].set_ylabel("Temperature (°C)")
axs[0, 0].legend()
axs[0, 0].grid(True)

# 2. Rainfall Bar Chart
axs[0, 1].bar(grouped["year"], grouped["rainfall"], color='blue', label='Rainfall (mm)')
axs[0, 1].set_title("Yearly Rainfall Distribution")
axs[0, 1].set_xlabel("Year")
axs[0, 1].set_ylabel("Rainfall (mm)")
axs[0, 1].legend()
axs[0, 1].grid(True)

# 3. Scatter Plot – Humidity vs Temperature
axs[1, 0].scatter(df["temperature"], df["humidity"], color='green', alpha=0.6)
axs[1, 0].set_title("Humidity vs Temperature Correlation")
axs[1, 0].set_xlabel("Temperature (°C)")
axs[1, 0].set_ylabel("Humidity (%)")
axs[1, 0].grid(True)

# 4. Linear Regression Plot
axs[1, 1].scatter(grouped["year"], grouped["temperature"], color='purple', label="Actual Temperature")
axs[1, 1].plot(grouped["year"], predicted, color='orange', linestyle='--', linewidth=2, label="Predicted Trend")
axs[1, 1].set_title("Temperature Prediction for Next Years")
axs[1, 1].set_xlabel("Year")
axs[1, 1].set_ylabel("Temperature (°C)")
axs[1, 1].legend()
axs[1, 1].grid(True)

# Show all 4 plots
plt.show()
