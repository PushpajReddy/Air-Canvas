import cv2
import numpy as np
import time
import os
from datetime import datetime
from collections import deque
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

try:
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

import mediapipe as mp

print("─" * 50)
print(f"OpenCV    {cv2.__version__}")
print(f"MediaPipe {mp.__version__}")
print(f"OCR       {'available' if OCR_AVAILABLE else 'not installed'}")
print(f"PDF       {'available' if PDF_AVAILABLE else 'install pdf2image + poppler'}")
print("─" * 50 + "\n")


class KalmanFilter2D:
    def __init__(self, process_noise=0.1, measure_noise=1.0):
        def make_kf():
            kf = cv2.KalmanFilter(2, 1)
            kf.measurementMatrix   = np.array([[1, 0]], np.float32)
            kf.transitionMatrix    = np.array([[1, 1], [0, 1]], np.float32)
            kf.processNoiseCov     = np.eye(2, dtype=np.float32) * process_noise
            kf.measurementNoiseCov = np.array([[measure_noise]], np.float32)
            return kf
        self.kf_x = make_kf()
        self.kf_y = make_kf()
        self.initialized = False

    def update(self, x, y):
        if not self.initialized:
            for kf, v in [(self.kf_x, x), (self.kf_y, y)]:
                kf.statePre  = np.array([[np.float32(v)], [0]], np.float32)
                kf.statePost = np.array([[np.float32(v)], [0]], np.float32)
            self.initialized = True
        self.kf_x.predict()
        self.kf_y.predict()
        sx = self.kf_x.correct(np.array([[np.float32(x)]]))
        sy = self.kf_y.correct(np.array([[np.float32(y)]]))
        return int(sx[0][0]), int(sy[0][0])

    def reset(self):
        self.initialized = False


class HandDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.7):
        self.mpHands = mp.solutions.hands
        self.mpDraw  = mp.solutions.drawing_utils
        self.hands   = self.mpHands.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.results = None
        self.lmList  = []
        self._ema    = {}
        self._alpha  = 0.55

    def findHands(self, img, draw=True):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        if self.results.multi_hand_landmarks and draw:
            for lms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, lms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=False):
        self.lmList = []
        bbox = []
        if not self.results or not self.results.multi_hand_landmarks:
            self._ema.pop(handNo, None)
            return self.lmList, bbox
        if handNo >= len(self.results.multi_hand_landmarks):
            return self.lmList, bbox

        h, w, _ = img.shape
        raw = []
        for id, lm in enumerate(self.results.multi_hand_landmarks[handNo].landmark):
            raw.append([id, int(lm.x * w), int(lm.y * h)])

        if handNo not in self._ema or len(self._ema[handNo]) != len(raw):
            self._ema[handNo] = [list(r) for r in raw]
        else:
            a = self._alpha
            for i in range(len(raw)):
                self._ema[handNo][i][1] = int(a * raw[i][1] + (1-a) * self._ema[handNo][i][1])
                self._ema[handNo][i][2] = int(a * raw[i][2] + (1-a) * self._ema[handNo][i][2])

        self.lmList = [list(lm) for lm in self._ema[handNo]]
        xs = [lm[1] for lm in self.lmList]
        ys = [lm[2] for lm in self.lmList]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        if draw:
            for lm in self.lmList:
                cv2.circle(img, (lm[1], lm[2]), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList, bbox

    def fingersUp(self):
        if not self.lmList or len(self.lmList) < 21:
            return [0, 0, 0, 0, 0]
        try:
            lm = self.lmList
            wrist   = np.array([lm[0][1], lm[0][2]], dtype=float)
            mid_mcp = np.array([lm[9][1], lm[9][2]], dtype=float)
            axis = mid_mcp - wrist
            n = np.linalg.norm(axis)
            if n < 1e-3:
                return [0, 0, 0, 0, 0]
            axis /= n
            perp = np.array([-axis[1], axis[0]])
            fingers = []
            t_tip = np.array([lm[4][1], lm[4][2]], dtype=float)
            t_ip  = np.array([lm[3][1], lm[3][2]], dtype=float)
            fingers.append(1 if np.dot(t_tip - t_ip, perp) > 0 else 0)
            for tid in [8, 12, 16, 20]:
                tip = np.array([lm[tid][1],   lm[tid][2]],   dtype=float)
                pip = np.array([lm[tid-2][1], lm[tid-2][2]], dtype=float)
                fingers.append(1 if np.dot(tip - pip, axis) > 0 else 0)
            return fingers
        except Exception:
            return [0, 0, 0, 0, 0]


class VirtualPainter:
    WIDTH    = 1280
    HEIGHT   = 720
    HEADER_H = 100
    COLORS = {
        'purple': (255, 0, 255),
        'blue':   (255, 0, 0),
        'green':  (0, 255, 0),
        'red':    (0, 0, 255),
        'yellow': (0, 255, 255),
        'black':  (0, 0, 0),
    }

    def __init__(self, camera_index=0):
        print("Initializing AirCanvas...")
        self.current_color   = self.COLORS['purple']
        self.brushThickness  = 15
        self.eraserThickness = 80
        self.xp, self.yp     = 0, 0
        self.imgCanvas = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)

        # Background image
        self.bg_image   = None
        self.bg_opacity = 0.6
        self.bg_enabled = False

        self._mode_buffer = deque(maxlen=5)
        self.kalman = KalmanFilter2D(process_noise=0.1, measure_noise=1.0)
        self.is_recording         = False
        self.show_help            = False
        self.video_writer         = None
        self.recording_start_time = None
        self.last_gesture_time    = 0
        self.gesture_cooldown     = 1.0

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            for alt in [1, 2]:
                self.cap = cv2.VideoCapture(alt)
                if self.cap.isOpened():
                    break
        if not self.cap.isOpened():
            raise RuntimeError("No camera found.")
        self.cap.set(3, self.WIDTH)
        self.cap.set(4, self.HEIGHT)

        self.detector = HandDetector(maxHands=1, detectionCon=0.5, trackCon=0.7)
        os.makedirs('saved_drawings', exist_ok=True)
        os.makedirs('recordings',     exist_ok=True)
        os.makedirs('extracted_text', exist_ok=True)
        print("AirCanvas ready!  O=open file  H=help\n")

    def load_background_file(self):
        if TKINTER_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askopenfilename(
                title="Select image or PDF to annotate",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"),
                    ("PDF files", "*.pdf"),
                    ("All files", "*.*"),
                ]
            )
            root.destroy()
        else:
            print("Enter full file path:")
            path = input("  > ").strip().strip('"')

        if path:
            self._load_file_from_path(path)
        else:
            print("No file selected")

    def _load_file_from_path(self, path):
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
        ext = os.path.splitext(path)[1].lower()
        if ext == '.pdf':
            if not PDF_AVAILABLE:
                print("PDF not supported. Run: pip install pdf2image")
                print("Also install Poppler: https://github.com/oschwartz10612/poppler-windows/releases")
                return
            try:
                pages = convert_from_path(path, dpi=150)
                frame = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
                print(f"PDF loaded ({len(pages)} pages) - showing page 1")
            except Exception as e:
                print(f"PDF error: {e}")
                return
        else:
            frame = cv2.imread(path)
            if frame is None:
                print(f"Could not read: {path}")
                return

        self.bg_image   = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
        self.bg_enabled = True
        self.imgCanvas  = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
        print(f"Loaded: {os.path.basename(path)}")
        print("  B = toggle image on/off   [ / ] = adjust opacity")

    def toggle_background(self):
        if self.bg_image is None:
            print("No image loaded - press O to open one")
        else:
            self.bg_enabled = not self.bg_enabled
            print(f"Background {'ON' if self.bg_enabled else 'OFF'}")

    def adjust_bg_opacity(self, delta):
        self.bg_opacity = max(0.1, min(1.0, self.bg_opacity + delta))
        print(f"Opacity: {int(self.bg_opacity*100)}%")

    def build_frame(self, cam_img):
        result = cam_img.copy()
        if self.bg_enabled and self.bg_image is not None:
            result = cv2.addWeighted(self.bg_image, self.bg_opacity,
                                     result, 1.0 - self.bg_opacity, 0)
        gray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(result, mask_inv)
        result = cv2.bitwise_or(result, self.imgCanvas)
        return result

    def fill_stroke(self, canvas, x0, y0, x1, y1, color, thickness):
        dist = np.hypot(x1-x0, y1-y0)
        if dist > 4:
            steps = max(1, int(dist // 4))
            for i in range(steps+1):
                t = i / steps
                cv2.circle(canvas,
                           (int(x0+t*(x1-x0)), int(y0+t*(y1-y0))),
                           thickness//2, color, -1)
        cv2.line(canvas, (x0, y0), (x1, y1), color, thickness)

    def reset_stroke(self):
        self.kalman.reset()
        self.xp, self.yp = 0, 0

    def get_stable_mode(self, fingers):
        if   fingers[1] == 1 and fingers[2] == 1: self._mode_buffer.append('select')
        elif fingers[1] == 1 and fingers[2] == 0: self._mode_buffer.append('draw')
        elif all(f == 1 for f in fingers):         self._mode_buffer.append('clear')
        else:                                       self._mode_buffer.append('idle')
        if (len(self._mode_buffer) == self._mode_buffer.maxlen and
                self._mode_buffer.count(self._mode_buffer[-1]) >= 3):
            return self._mode_buffer[-1]
        return 'idle'

    def process_gestures(self, fingers):
        # now = time.time()
        # if now - self.last_gesture_time < self.gesture_cooldown:
        #     return
        # if fingers == [1, 0, 0, 0, 0]:
        #     self.brushThickness = min(50, self.brushThickness + 5)
        #     self.last_gesture_time = now
        #     print(f"Brush: {self.brushThickness}")
        # elif fingers == [0, 1, 0, 0, 0]:
        #     self.brushThickness = max(5, self.brushThickness - 5)
        #     self.last_gesture_time = now
        #     print(f"Brush: {self.brushThickness}")
        pass

    def draw_ui(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (self.WIDTH, self.HEADER_H), (40, 40, 40), -1)
        for name, color, x in [
            ('Purple', self.COLORS['purple'], 50),
            ('Blue',   self.COLORS['blue'],  180),
            ('Green',  self.COLORS['green'], 310),
            ('Red',    self.COLORS['red'],   440),
            ('Yellow', self.COLORS['yellow'],570),
            ('Eraser', self.COLORS['black'], 700),
        ]:
            active = (color == self.current_color)
            cv2.circle(overlay, (x, 50), 25, color, -1 if active else 3)
            if active:
                cv2.circle(overlay, (x, 50), 32, (255, 255, 255), 2)
            cv2.putText(overlay, name, (x-30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        bx = 850
        cv2.putText(overlay, f"Size:{self.brushThickness}", (bx, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.rectangle(overlay, (bx, 45), (bx+100, 65), (60, 60, 60), -1)
        cv2.rectangle(overlay, (bx, 45), (bx+min(100, self.brushThickness*2), 65),
                      self.current_color, -1)
        if self.bg_image is not None:
            lbl = f"IMG {'ON' if self.bg_enabled else 'OFF'} {int(self.bg_opacity*100)}%"
            cv2.putText(overlay, lbl, (bx, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 200) if self.bg_enabled else (120, 120, 120), 1)
        if self.is_recording:
            cv2.circle(overlay, (bx+150, 50), 15, (0, 0, 255), -1)
            cv2.putText(overlay, "REC", (bx+175, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(overlay, (1050, 30), (1150, 70), (0, 200, 0), -1)
        cv2.putText(overlay, "SAVE", (1065, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(overlay, (1200, 50), 20, (100, 100, 255), -1)
        cv2.putText(overlay, "?", (1193, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        if self.show_help:
            self.draw_help(img)
        return img

    def draw_help(self, img):
        overlay = img.copy()
        cv2.rectangle(overlay, (180, 130), (1100, 600), (25, 25, 25), -1)
        cv2.rectangle(overlay, (180, 130), (1100, 600), (100, 200, 255), 3)
        lines = [
            "GESTURE CONTROLS:",
            "  Index + Middle up   ->  Selection / Color pick",
            "  Index only          ->  Drawing mode",
            "  All fingers up      ->  Clear canvas",
            "  Thumbs up           ->  Increase brush size",
            "",
            "KEYBOARD SHORTCUTS:",
            "  O        Open image or PDF to draw on top of",
            "  B        Toggle background image on / off",
            "  [ / ]    Decrease / Increase image opacity",
            "  S        Save drawing (background + your strokes)",
            "  C        Clear canvas strokes only",
            "  T        Extract handwritten text (OCR)",
            "  R        Start / Stop session recording",
            "  + / -    Adjust brush size",
            "  H        Toggle this help panel",
            "  ESC      Exit",
        ]
        y = 160
        for line in lines:
            is_hdr = line.endswith(':')
            cv2.putText(overlay, line, (210, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65 if is_hdr else 0.52,
                        (100, 200, 255) if is_hdr else (255, 255, 255), 1)
            y += 25
        cv2.addWeighted(overlay, 0.93, img, 0.07, 0, img)

    def save_drawing(self):
        try:
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn  = f"saved_drawings/drawing_{ts}.png"
            bg  = self.bg_image if (self.bg_enabled and self.bg_image is not None) \
                  else np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
            cv2.imwrite(fn, self.build_frame(bg))
            print(f"Saved: {fn}")
        except Exception as e:
            print(f"Save error: {e}")

    def toggle_recording(self, img):
        try:
            if not self.is_recording:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"recordings/session_{ts}.avi"
                self.video_writer = cv2.VideoWriter(
                    fn, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (self.WIDTH, self.HEIGHT))
                self.is_recording = True
                self.recording_start_time = time.time()
                print(f"Recording: {fn}")
            else:
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                self.is_recording = False
                print(f"Stopped ({time.time()-self.recording_start_time:.1f}s)")
        except Exception as e:
            print(f"Recording error: {e}")
            self.is_recording = False

    def extract_text_from_canvas(self):
        if not OCR_AVAILABLE:
            print("pip install pytesseract pillow")
            return
        try:
            gray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            th = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
            pad = cv2.copyMakeBorder(th, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
            pil = Image.fromarray(pad)
            best = ""
            for psm in [11, 6, 3, 7, 8]:
                r = pytesseract.image_to_string(pil, config=f"--psm {psm} --oem 1").strip()
                if r and len(r) > len(best):
                    best = r
            if best:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = f"extracted_text/text_{ts}.txt"
                open(fn, 'w', encoding='utf-8').write(best)
                print(f"Text saved: {fn}\n{best}")
            else:
                print("No text detected - write larger, use bright color, increase brush (+)")
                Image.fromarray(pad).save("extracted_text/debug_ocr_input.png")
        except Exception as e:
            print(f"OCR error: {e}")

    def run(self):
        print("Running - press H for help\n")
        try:
            while True:
                ok, cam = self.cap.read()
                if not ok:
                    break
                cam = cv2.flip(cam, 1)
                cam = self.detector.findHands(cam, draw=False)
                lmList, _ = self.detector.findPosition(cam, draw=False)

                if lmList:
                    x1, y1 = lmList[8][1],  lmList[8][2]
                    x2, y2 = lmList[12][1], lmList[12][2]
                    fingers = self.detector.fingersUp()
                    self.process_gestures(fingers)
                    mode = self.get_stable_mode(fingers)

                    if mode == 'select':
                        self.reset_stroke()
                        if y1 < self.HEADER_H:
                            if   25  < x1 < 75:    self.current_color = self.COLORS['purple']
                            elif 155 < x1 < 205:   self.current_color = self.COLORS['blue']
                            elif 285 < x1 < 335:   self.current_color = self.COLORS['green']
                            elif 415 < x1 < 465:   self.current_color = self.COLORS['red']
                            elif 545 < x1 < 595:   self.current_color = self.COLORS['yellow']
                            elif 675 < x1 < 725:   self.current_color = self.COLORS['black']
                            elif 1050 < x1 < 1150: self.save_drawing()
                            elif 1180 < x1 < 1220: self.show_help = not self.show_help
                        cv2.rectangle(cam, (x1, y1-15), (x2, y2+15),
                                      self.current_color, cv2.FILLED)

                    elif mode == 'draw':
                        sx, sy = self.kalman.update(x1, y1)
                        cv2.circle(cam, (sx, sy), 10, self.current_color, cv2.FILLED)
                        if self.xp == 0 and self.yp == 0:
                            self.xp, self.yp = sx, sy
                        thick = (self.eraserThickness
                                 if self.current_color == self.COLORS['black']
                                 else self.brushThickness)
                        self.fill_stroke(cam,            self.xp, self.yp, sx, sy, self.current_color, thick)
                        self.fill_stroke(self.imgCanvas, self.xp, self.yp, sx, sy, self.current_color, thick)
                        self.xp, self.yp = sx, sy

                    elif mode == 'clear':
                        self.imgCanvas = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
                        self.reset_stroke()

                    else:
                        self.reset_stroke()
                else:
                    self.reset_stroke()
                    self._mode_buffer.clear()

                frame = self.build_frame(cam)
                frame = self.draw_ui(frame)

                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)

                cv2.imshow("AirCanvas", frame)
                key = cv2.waitKey(1) & 0xFF

                if   key == 27:                    break
                elif key in (ord('s'), ord('S')):  self.save_drawing()
                elif key in (ord('r'), ord('R')):  self.toggle_recording(frame)
                elif key in (ord('t'), ord('T')):  self.extract_text_from_canvas()
                elif key in (ord('c'), ord('C')):
                    self.imgCanvas = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
                    print("Canvas cleared")
                elif key in (ord('h'), ord('H')):  self.show_help = not self.show_help
                elif key in (ord('o'), ord('O')):  self.load_background_file()
                elif key in (ord('b'), ord('B')):  self.toggle_background()
                elif key == ord(']'):              self.adjust_bg_opacity(+0.1)
                elif key == ord('['):              self.adjust_bg_opacity(-0.1)
                elif key in (ord('+'), ord('=')):
                    self.brushThickness = min(50, self.brushThickness + 2)
                    print(f"Brush: {self.brushThickness}")
                elif key in (ord('-'), ord('_')):
                    self.brushThickness = max(5, self.brushThickness - 2)
                    print(f"Brush: {self.brushThickness}")

        except KeyboardInterrupt:
            print("Interrupted")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.is_recording and self.video_writer:
                self.video_writer.release()
            self.cap.release()
            cv2.destroyAllWindows()
            print("Done")


if __name__ == "__main__":
    try:
        VirtualPainter(camera_index=0).run()
    except Exception as e:
        print(f"Fatal: {e}")
        import traceback
        traceback.print_exc()
