import pandas as pd

# Read dataset
df = pd.read_csv("data/landmarks.csv", header=None)

print("\nDataset Summary")
print("-" * 30)

# Total samples
print("Total Samples:", len(df))

# Shape information
print("\nShape:", df.shape)

print("\nMeaning:")
print(f"{df.shape[0]} rows")
print(f"{df.shape[1]} columns")

print("\nColumn Breakdown:")
print("1 label column")
print(f"{df.shape[1] - 1} feature columns")

# Class counts
print("\nClass Counts:")
counts = df[0].value_counts()

for label, count in counts.items():
    print(f"{label}: {count}")