from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("data/landmarks.csv", header=None)

# Features and labels
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

# Save model
with open("model/asl_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save label encoder
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\nModel and encoder saved successfully!")