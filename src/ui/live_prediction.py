import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load model
with open("model/asl_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoder
with open("model/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    prediction_text = "No Hand"

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == 63:

            X = np.array(landmarks).reshape(1, -1)

            prediction = model.predict(X)

            predicted_label = encoder.inverse_transform(prediction)[0]

            prediction_text = predicted_label

    cv2.putText(
        frame,
        f"Prediction: {prediction_text}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Live Prediction", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()