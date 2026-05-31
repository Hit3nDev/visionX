import cv2
import mediapipe as mp

class HandTracker:

    def __init__(self):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def get_screen_points(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        if len(results.multi_hand_landmarks) < 2:
            return None

        h,w,_ = frame.shape

        hands_data = []

        for hand in results.multi_hand_landmarks:

            thumb = hand.landmark[4]
            index = hand.landmark[8]
            wrist = hand.landmark[0]

            hands_data.append({
                "thumb": (
                    int(thumb.x*w),
                    int(thumb.y*h)
                ),
                "index": (
                    int(index.x*w),
                    int(index.y*h)
                ),
                "wrist": (
                    int(wrist.x*w),
                    int(wrist.y*h)
                )
            })

        left = hands_data[0]
        right = hands_data[1]

        lt = left["index"]
        lb = left["thumb"]

        rt = right["index"]
        rb = right["thumb"]

        return [lt, rt, rb, lb]