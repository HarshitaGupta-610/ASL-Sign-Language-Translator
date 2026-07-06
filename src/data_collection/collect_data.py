import cv2
import mediapipe as mp
import csv
import os

LETTER = input("Enter letter to collect: ").upper()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

os.makedirs("data", exist_ok=True)

cap = cv2.VideoCapture(0)

sample_count = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        cv2.putText(
            frame,
            f"Letter: {LETTER}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        
    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:
     print("Key pressed:", key)

    if key == ord('s'):

        if not results.multi_hand_landmarks:
            print("❌ No hand detected. Show your hand and press S again.")

        else:
            hand = results.multi_hand_landmarks[0]

            row = [LETTER]

            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])

            with open("data/landmarks.csv", "a", newline="") as f:
                csv.writer(f).writerow(row)

            sample_count += 1
            print(f"✅ Saved sample {sample_count}")

    if key == 27:
        break



cap.release()
cv2.destroyAllWindows()