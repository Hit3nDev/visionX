import cv2
import mediapipe as mp

class HandTracker:

    def __init__(self):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def get_data(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        h, w, _ = frame.shape

        hands = []

        for hand in results.multi_hand_landmarks:

            lm = hand.landmark

            hands.append({
                "thumb": (
                    int(lm[4].x * w),
                    int(lm[4].y * h)
                ),
                "index": (
                    int(lm[8].x * w),
                    int(lm[8].y * h)
                ),
                "middle": (
                    int(lm[12].x * w),
                    int(lm[12].y * h)
                ),
                "ring": (
                    int(lm[16].x * w),
                    int(lm[16].y * h)
                ),
                "pinky": (
                    int(lm[20].x * w),
                    int(lm[20].y * h)
                ),
                "landmarks": lm
            })

        return hands

    def detect_mode(self, hand):

        lm = hand["landmarks"]

        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up = lm[16].y < lm[14].y

        if index_up and not middle_up:
            return 1

        if middle_up and not ring_up:
            return 2

        if ring_up:
            return 3

        return 0