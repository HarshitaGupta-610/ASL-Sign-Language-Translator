import pandas as pd
import matplotlib.pyplot as plt

# Read dataset
df = pd.read_csv("data/landmarks.csv", header=None)

# Take first sample
sample = df.iloc[0]

# First column is label
label = sample[0]

# Remaining 63 values are landmarks
coords = sample[1:].values.astype(float)

# Reshape into 21 landmarks × 3 coordinates
landmarks = coords.reshape(21, 3)

# Plot landmarks
plt.figure(figsize=(6, 6))

for i, (x, y, z) in enumerate(landmarks):
    plt.scatter(x, y, color="blue")
    plt.text(x, y, str(i), fontsize=8)

plt.title(f"Landmark Visualization - Label: {label}")
plt.xlabel("X")
plt.ylabel("Y")

# Invert Y axis to match image coordinates
plt.gca().invert_yaxis()

plt.grid(True)
plt.show()