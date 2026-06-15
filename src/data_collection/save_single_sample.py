import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        for lm in hand.landmark:
            cv2.circle(
                frame,
                (int(lm.x * frame.shape[1]),
                 int(lm.y * frame.shape[0])),
                5,
                (0,255,0),
                -1
            )

    cv2.imshow("Save Sample", frame)

    key = cv2.waitKey(1)

    if key == ord('s') and results.multi_hand_landmarks:

        row = []

        for lm in hand.landmark:
            row.extend([lm.x, lm.y, lm.z])

        with open("data/sample.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)

        print("Sample Saved!")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()