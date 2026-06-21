import pandas as pd
import pickle

# Load dataset
df = pd.read_csv("data/landmarks.csv", header=None)

print("=" * 40)
print("MODEL EVALUATION")
print("=" * 40)

print("\nDataset Shape:")
print(df.shape)

print("\nClasses Present:")
print(df[0].unique())

# Load model
with open("model/asl_model.pkl", "rb") as f:
    model = pickle.load(f)

print("\nModel Loaded Successfully!")

# Load encoder
with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

print("Encoder Loaded Successfully!")

print("\nClass Mapping:")

for i, label in enumerate(encoder.classes_):
    print(f"{label} -> {i}")