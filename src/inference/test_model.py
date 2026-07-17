import pandas as pd
import pickle

# Load dataset
df = pd.read_csv("data/landmarks.csv", header=None)

# Take first sample
sample = df.iloc[0]

actual_label = sample[0]

X = sample[1:].values.reshape(1, -1)

# Load model
with open("model/asl_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoder
with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Predict
prediction = model.predict(X)

print("Raw Prediction:", prediction[0])

predicted_label = encoder.inverse_transform(prediction)[0]

print("Actual Label:", actual_label)
print("Predicted Label:", predicted_label)