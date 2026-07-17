import cv2
import mediapipe as mp

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

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Get frame dimensions
            h, w, _ = frame.shape

            # Draw landmark numbers
            for idx, lm in enumerate(hand_landmarks.landmark):

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.putText(
                    frame,
                    str(idx),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

    cv2.imshow("Hand Visualizer", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()