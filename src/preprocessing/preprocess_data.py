import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/landmarks.csv", header=None)

print("Dataset Shape:")
print(df.shape)

# Features
X = df.iloc[:, 1:]

# Labels
y = df.iloc[:, 0]

print("\nFeature Shape:")
print(X.shape)

print("\nLabels:")
print(y.unique())

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print("\nClass Mapping:")

for label, encoded in zip(encoder.classes_, range(len(encoder.classes_))):
    print(f"{label} -> {encoded}")

print("\nEncoded Labels:")
print(y_encoded[:10])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:")
print(X_train.shape)

print("\nTest Shape:")
print(X_test.shape)