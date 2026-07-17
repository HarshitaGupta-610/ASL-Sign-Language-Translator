import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/landmarks.csv", header=None)

print("=" * 40)
print("DATASET INFORMATION")
print("=" * 40)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nClass Counts:")
print(df[0].value_counts())

# Class Distribution Graph
df[0].value_counts().plot(
    kind="bar",
    color=["blue", "green", "orange"]
)

plt.title("ASL Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Samples")
plt.grid(True)

plt.show()