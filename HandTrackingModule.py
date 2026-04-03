import cv2
import mediapipe as mp
import math
import numpy as np
import time


class handDetector():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None
        self.lmList = []

        # ── Per-landmark EMA smoothing ─────────────────────────────────────
        # alpha=1.0 = raw (no smoothing), alpha=0.4 = heavy smoothing
        # 0.55 gives fluid tracking without visible lag
        self._ema_alpha = 0.55
        self._ema_lm = {}   # {hand_index: [[id, x, y], ...]}

    # ───────────────────────────────────────────────────────────────────────
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    # ───────────────────────────────────────────────────────────────────────
    def findPosition(self, img, handNo=0, draw=False):
        """
        Returns EMA-smoothed landmark list and bounding box.
        EMA per-landmark kills single-frame jitter without adding lag.
        """
        self.lmList = []
        bbox = []

        if not self.results or not self.results.multi_hand_landmarks:
            self._ema_lm.pop(handNo, None)   # reset EMA when hand leaves frame
            return self.lmList, bbox

        if handNo >= len(self.results.multi_hand_landmarks):
            return self.lmList, bbox

        myHand = self.results.multi_hand_landmarks[handNo]
        h, w, _ = img.shape

        raw = []
        for id, lm in enumerate(myHand.landmark):
            raw.append([id, int(lm.x * w), int(lm.y * h)])

        # Apply EMA smoothing
        if handNo not in self._ema_lm or len(self._ema_lm[handNo]) != len(raw):
            self._ema_lm[handNo] = [list(r) for r in raw]   # seed with first frame
        else:
            a = self._ema_alpha
            for i in range(len(raw)):
                self._ema_lm[handNo][i][1] = int(
                    a * raw[i][1] + (1 - a) * self._ema_lm[handNo][i][1])
                self._ema_lm[handNo][i][2] = int(
                    a * raw[i][2] + (1 - a) * self._ema_lm[handNo][i][2])

        self.lmList = [list(lm) for lm in self._ema_lm[handNo]]

        xs = [lm[1] for lm in self.lmList]
        ys = [lm[2] for lm in self.lmList]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        if draw:
            for lm in self.lmList:
                cv2.circle(img, (lm[1], lm[2]), 5, (255, 0, 255), cv2.FILLED)
            cv2.rectangle(img, (bbox[0]-20, bbox[1]-20),
                          (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)

        return self.lmList, bbox

    # ───────────────────────────────────────────────────────────────────────
    def fingersUp(self):
        """
        Axis-relative finger detection.

        The original code compared tip.y < pip.y in raw screen space.
        This breaks when your hand tilts — a tilted index finger reads as
        'down' even when fully extended, cutting the drawing stroke.

        Fix: project each finger's tip-pip vector onto the hand's own
        wrist-to-middle-MCP axis. Positive dot product = finger extended,
        regardless of hand rotation or tilt.
        """
        fingers = []

        if not self.lmList or len(self.lmList) < 21:
            return [0, 0, 0, 0, 0]

        try:
            lm = self.lmList

            # Hand axis: wrist (0) -> middle-finger MCP (9)
            wrist    = np.array([lm[0][1],  lm[0][2]],  dtype=float)
            mid_mcp  = np.array([lm[9][1],  lm[9][2]],  dtype=float)
            hand_axis = mid_mcp - wrist
            axis_len  = np.linalg.norm(hand_axis)
            if axis_len < 1e-3:
                return [0, 0, 0, 0, 0]
            hand_axis /= axis_len   # normalize to unit vector

            # Thumb moves sideways, so check against perpendicular axis
            perp = np.array([-hand_axis[1], hand_axis[0]])   # 90 deg CCW
            t_tip = np.array([lm[4][1], lm[4][2]], dtype=float)
            t_ip  = np.array([lm[3][1], lm[3][2]], dtype=float)
            fingers.append(1 if np.dot(t_tip - t_ip, perp) > 0 else 0)

            # Index, Middle, Ring, Pinky
            for tip_id in [8, 12, 16, 20]:
                tip = np.array([lm[tip_id][1],     lm[tip_id][2]],     dtype=float)
                pip = np.array([lm[tip_id - 2][1], lm[tip_id - 2][2]], dtype=float)
                fingers.append(1 if np.dot(tip - pip, hand_axis) > 0 else 0)

        except Exception:
            return [0, 0, 0, 0, 0]

        return fingers

    # ───────────────────────────────────────────────────────────────────────
    def findDistance(self, p1, p2, img=None, draw=True, r=15, t=3):
        length = 0
        if not self.lmList:
            return length, img, [0, 0, 0, 0, 0, 0]

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)

        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]


# ── Standalone test ──────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    pTime = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=True)
        if lmList:
            fingers = detector.fingersUp()
            cv2.putText(img, str(fingers), (10, 110),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS:{int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Hand Tracker", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
