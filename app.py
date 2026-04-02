# EnhancedVirtualPainter.py - Universal MediaPipe Compatibility
import cv2
import numpy as np
import time
import os
import sys
from datetime import datetime
from collections import deque
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:/Program Files/Tesseract-OCR/tesseract.exe"
)

MEDIAPIPE_NEW_API = False

# Comprehensive dependency checking with MediaPipe version compatibility
def check_dependencies():
    """Check and validate all required dependencies"""
    errors = []
    
    # Check OpenCV
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError:
        errors.append("opencv-python")
    
    # Check MediaPipe with version compatibility
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe version: {mp.__version__}")
        
        # Try to access solutions - compatible with both old and new versions
        try:
            # New API (0.10+)
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            print("  → Using MediaPipe Tasks API (0.10+)")
        except ImportError:
            # Old API (pre-0.10)
            if hasattr(mp, 'solutions'):
                print("  → Using MediaPipe Solutions API (pre-0.10)")
            else:
                errors.append("mediapipe (incompatible version)")
    except ImportError:
        errors.append("mediapipe")
    except Exception as e:
        errors.append(f"mediapipe (error: {e})")
    
    # Check NumPy
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        errors.append("numpy")
    
    if errors:
        print("\n❌ Missing or corrupted dependencies detected:")
        for err in errors:
            print(f"   - {err}")
        print("\n🔧 Fix with:")
        print("   pip uninstall mediapipe opencv-python")
        print("   pip install mediapipe==0.10.9 opencv-python numpy")
        sys.exit(1)
    
    print("✓ All core dependencies OK\n")

check_dependencies()

# Import MediaPipe with compatibility layer
import mediapipe as mp

# Try new API first, fall back to old API
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe.framework.formats import landmark_pb2
    MEDIAPIPE_NEW_API = True
    print("✓ Using MediaPipe Tasks API (v0.10+)\n")
except ImportError:
    MEDIAPIPE_NEW_API = False
    print("✓ Using MediaPipe Solutions API\n")

# Optional OCR dependencies
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
    print("✓ OCR libraries available")
except ImportError:
    OCR_AVAILABLE = False
    print("⚠ OCR not available (optional)")

print("-" * 50 + "\n")


class HandDetector:
    """Universal hand detection compatible with all MediaPipe versions"""
    
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        
        try:
            if MEDIAPIPE_NEW_API:
                # New API (MediaPipe 0.10+)
                base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=self.max_hands,
                    min_hand_detection_confidence=self.detection_con,
                    min_tracking_confidence=self.track_con
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.mp_hands = None
                self.mp_draw = None
            else:
                # Old API (MediaPipe pre-0.10)
                self.mp_hands = mp.solutions.hands
                self.mp_draw = mp.solutions.drawing_utils
                self.hands = self.mp_hands.Hands(
                    static_image_mode=self.mode,
                    max_num_hands=self.max_hands,
                    min_detection_confidence=self.detection_con,
                    min_tracking_confidence=self.track_con
                )
                self.detector = None
            
            self.tip_ids = [4, 8, 12, 16, 20]
            self.results = None
            self.lm_list = []
            
            print("✓ Hand detector initialized successfully")
            
        except Exception as e:
            # Fallback: Try old API even if new API was detected
            try:
                print(f"⚠ New API failed ({e}), trying old API...")
                self.mp_hands = mp.solutions.hands
                self.mp_draw = mp.solutions.drawing_utils
                self.hands = self.mp_hands.Hands(
                    static_image_mode=self.mode,
                    max_num_hands=self.max_hands,
                    min_detection_confidence=self.detection_con,
                    min_tracking_confidence=self.track_con
                )
                self.detector = None
                print("✓ Hand detector initialized with old API")
            except Exception as e2:
                print(f"❌ Failed to initialize hand detector: {e2}")
                raise
    
    def find_hands(self, img, draw=True):
        """Detect hands in image - compatible with both APIs"""
        try:
            if MEDIAPIPE_NEW_API and self.detector:
                # New API processing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                timestamp_ms = int(time.time() * 1000)
                self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
                print("Hands detected:",len(self.results.hand_landmarks) if self.results.hand_landmarks else 0)

                
                # Draw landmarks if requested
                if draw and self.results.hand_landmarks:
                    for hand_landmarks in self.results.hand_landmarks:
                        # Draw connections manually for new API
                        for connection in mp.solutions.hands.HAND_CONNECTIONS:
                            start_idx = connection[0]
                            end_idx = connection[1]
                            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                                start = hand_landmarks[start_idx]
                                end = hand_landmarks[end_idx]
                                h, w, _ = img.shape
                                start_point = (int(start.x * w), int(start.y * h))
                                end_point = (int(end.x * w), int(end.y * h))
                                cv2.line(img, start_point, end_point, (0, 255, 0), 2)
            else:
                # Old API processing
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.results = self.hands.process(img_rgb)
                
                if self.results.multi_hand_landmarks and draw:
                    for hand_lms in self.results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            img, hand_lms, self.mp_hands.HAND_CONNECTIONS
                        )
        except Exception as e:
            print(f"Warning: Hand detection error: {e}")
        
        return img
    
    # def find_position(self, img, hand_no=0, draw=False):
    #     """Get landmark positions - compatible with both APIs"""
    #     x_list, y_list = [], []
    #     bbox = []
    #     self.lm_list = []
        
    #     try:
    #         if MEDIAPIPE_NEW_API and self.detector:
    #             # New API
    #             if not self.results or not self.results.hand_landmarks:
    #                 return self.lm_list, bbox
                
    #             if hand_no >= len(self.results.hand_landmarks):
    #                 return self.lm_list, bbox
                
    #             hand_landmarks = self.results.hand_landmarks[hand_no]
    #             h, w, c = img.shape
                
    #             for id, lm in enumerate(hand_landmarks):
    #                 cx, cy = int(lm.x * w), int(lm.y * h)
    #                 x_list.append(cx)
    #                 y_list.append(cy)
    #                 self.lm_list.append([id, cx, cy])
                    
    #                 if draw:
    #                     cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    #         else:
    #             # Old API
    #             if not self.results or not self.results.multi_hand_landmarks:
    #                 return self.lm_list, bbox
                
    #             if hand_no >= len(self.results.multi_hand_landmarks):
    #                 return self.lm_list, bbox
                
    #             my_hand = self.results.multi_hand_landmarks[hand_no]
    #             h, w, c = img.shape
                
    #             for id, lm in enumerate(my_hand.landmark):
    #                 cx, cy = int(lm.x * w), int(lm.y * h)
    #                 x_list.append(cx)
    #                 y_list.append(cy)
    #                 self.lm_list.append([id, cx, cy])
                    
    #                 if draw:
    #                     cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
    #         if x_list and y_list:
    #             x_min, x_max = min(x_list), max(x_list)
    #             y_min, y_max = min(y_list), max(y_list)
    #             bbox = (x_min, y_min, x_max, y_max)
                
    #             if draw:
    #                 cv2.rectangle(img, (x_min - 20, y_min - 20),
    #                             (x_max + 20, y_max + 20), (0, 255, 0), 2)
        
    #     except Exception as e:
    #         print(f"Warning: Position detection error: {e}")
        
    #     return self.lm_list, bbox

    def find_position(self, img, hand_no=0, draw=False):
        x_list, y_list = [], []
        bbox = []
        new_lm_list = []

        try:
            if MEDIAPIPE_NEW_API and self.detector:
                if not self.results or not self.results.hand_landmarks:
                    return self.lm_list, bbox   # return LAST known — don't snap to empty
                if hand_no >= len(self.results.hand_landmarks):
                    return self.lm_list, bbox
                hand_landmarks = self.results.hand_landmarks[hand_no]
                h, w, c = img.shape
                for id, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    new_lm_list.append([id, cx, cy])
            else:
                if not self.results or not self.results.multi_hand_landmarks:
                    return self.lm_list, bbox   # return LAST known
                if hand_no >= len(self.results.multi_hand_landmarks):
                    return self.lm_list, bbox
                my_hand = self.results.multi_hand_landmarks[hand_no]
                h, w, c = img.shape
                for id, lm in enumerate(my_hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    new_lm_list.append([id, cx, cy])

            # Landmark stability — blend instead of snap on big jumps
            # Prevents jitter when hand rotates or partially occludes
            # BLEND = 0.6   # 0 = fully old, 1 = fully new
            # MAX_JUMP = 60 # pixels — beyond this, blend instead of hard-update
            # if self.lm_list and len(self.lm_list) == len(new_lm_list):
            #     for i in range(len(new_lm_list)):
            #         ox, oy = self.lm_list[i][1], self.lm_list[i][2]
            #         nx, ny = new_lm_list[i][1], new_lm_list[i][2]
            #         dist = np.hypot(nx - ox, ny - oy)
            #         if dist > MAX_JUMP:
            #             # Big jump — blend smoothly
            #             nx = int(ox + BLEND * (nx - ox))
            #             ny = int(oy + BLEND * (ny - oy))
            #         new_lm_list[i][1] = nx
            #         new_lm_list[i][2] = ny

            # self.lm_list = new_lm_list

            
            self.lm_list = new_lm_list

            for item in self.lm_list:
                x_list.append(item[1])
                y_list.append(item[2])
                if draw:
                    cv2.circle(img, (item[1], item[2]), 5, (255, 0, 255), cv2.FILLED)

            if x_list and y_list:
                bbox = (min(x_list), min(y_list), max(x_list), max(y_list))
                if draw:
                    cv2.rectangle(img, (bbox[0]-20, bbox[1]-20),
                                (bbox[2]+20, bbox[3]+20), (0,255,0), 2)

        except Exception as e:
            print(f"Warning: Position detection error: {e}")

        return self.lm_list, bbox
    
    # def fingers_up(self):
    #     """Detect which fingers are up"""
    #     fingers = []
        
    #     if not self.lm_list or len(self.lm_list) < 21:
    #         return [0, 0, 0, 0, 0]
        
    #     try:
    #         # Thumb (horizontal check)
    #         if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
    #             fingers.append(1)
    #         else:
    #             fingers.append(0)
            
    #         # Other fingers (vertical check)
    #         for id in range(1, 5):
    #             if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
    #                 fingers.append(1)
    #             else:
    #                 fingers.append(0)
        
    #     except Exception as e:
    #         return [0, 0, 0, 0, 0]
        
    #     return fingers

    def fingers_up(self):
        fingers = []

        if not self.lm_list or len(self.lm_list) < 21:
            return [0, 0, 0, 0, 0]

        try:
            # Wrist and middle finger MCP — defines the hand's own axis
            # This works regardless of how the hand is rotated/tilted
            wrist   = np.array([self.lm_list[0][1],  self.lm_list[0][2]])
            mid_mcp = np.array([self.lm_list[9][1],  self.lm_list[9][2]])
            hand_vec = mid_mcp - wrist  # points from wrist toward fingers

            # Thumb — check if tip is away from palm center (works at any angle)
            thumb_tip = np.array([self.lm_list[4][1],  self.lm_list[4][2]])
            thumb_ip  = np.array([self.lm_list[3][1],  self.lm_list[3][2]])
            thumb_vec = thumb_tip - thumb_ip
            # Cross product sign tells us which side the thumb is on
            cross = hand_vec[0] * thumb_vec[1] - hand_vec[1] * thumb_vec[0]
            fingers.append(1 if cross > 0 else 0)

            # Other 4 fingers — tip above its own PIP joint along hand axis
            for tip_id in [8, 12, 16, 20]:
                tip = np.array([self.lm_list[tip_id][1],     self.lm_list[tip_id][2]])
                pip = np.array([self.lm_list[tip_id - 2][1], self.lm_list[tip_id - 2][2]])
                # Project tip-to-pip vector onto hand axis
                tip_vec = tip - pip
                dot = np.dot(tip_vec, hand_vec)
                fingers.append(1 if dot > 0 else 0)

        except Exception:
            return [0, 0, 0, 0, 0]

        return fingers


class KalmanFilter2D:
    """
    2D Kalman Filter for finger tracking.
 
    Tracks X and Y independently. Each axis models:
      - position  (what we measure from MediaPipe)
      - velocity  (estimated from frame-to-frame change)
 
    This lets the filter PREDICT where the finger will be next frame,
    so fast movements produce smooth, gap-free strokes.
 
    Tuning knobs:
      process_noise  — higher = follows raw input more closely (less lag, more jitter)
      measure_noise  — higher = trusts MediaPipe less (more smoothing, more lag)
    """
 
    def __init__(self, process_noise=0.03, measure_noise=5.0):
        # ── X axis ────────────────────────────────────────────────────────────
        self.kf_x = cv2.KalmanFilter(2, 1)          # 2 states (pos, vel), 1 measurement
        self.kf_x.measurementMatrix  = np.array([[1, 0]], np.float32)
        self.kf_x.transitionMatrix   = np.array([[1, 1],
                                                  [0, 1]], np.float32)
        self.kf_x.processNoiseCov    = np.eye(2, dtype=np.float32) * process_noise
        self.kf_x.measurementNoiseCov = np.array([[measure_noise]], np.float32)
 
        # ── Y axis ────────────────────────────────────────────────────────────
        self.kf_y = cv2.KalmanFilter(2, 1)
        self.kf_y.measurementMatrix  = np.array([[1, 0]], np.float32)
        self.kf_y.transitionMatrix   = np.array([[1, 1],
                                                  [0, 1]], np.float32)
        self.kf_y.processNoiseCov    = np.eye(2, dtype=np.float32) * process_noise
        self.kf_y.measurementNoiseCov = np.array([[measure_noise]], np.float32)
 
        self.initialized = False
 
    def update(self, x, y):
        """Feed a raw MediaPipe point → get back a smoothed, predicted point."""
        mx = np.array([[np.float32(x)]])
        my = np.array([[np.float32(y)]])
 
        if not self.initialized:
            # Seed the filter with the first real measurement so it
            # doesn't start from (0,0) and snap across the screen
            self.kf_x.statePre  = np.array([[np.float32(x)], [0]], np.float32)
            self.kf_x.statePost = np.array([[np.float32(x)], [0]], np.float32)
            self.kf_y.statePre  = np.array([[np.float32(y)], [0]], np.float32)
            self.kf_y.statePost = np.array([[np.float32(y)], [0]], np.float32)
            self.initialized = True
 
        self.kf_x.predict()
        self.kf_y.predict()
 
        sx = self.kf_x.correct(mx)
        sy = self.kf_y.correct(my)
 
        return int(sx[0][0]), int(sy[0][0])
 
    def reset(self):
        """Call this when the finger lifts — resets velocity so next
        stroke starts fresh without carrying over old momentum."""
        self.initialized = False

class VirtualPainter:
    """Enhanced Virtual Painter Application"""
    
    def __init__(self, camera_index=0):
        print("🎨 Initializing Virtual Painter...")
        
        # Canvas settings
        self.WIDTH = 1280
        self.HEIGHT = 720
        self.HEADER_H = 100
        
        # Drawing parameters
        self.brush_thickness = 15
        self.eraser_thickness = 80
        self.smoothing_window = 7
        
        # Colors
        self.COLORS = {
            'purple': (255, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'black': (0, 0, 0)
        }
        
        # State
        self.current_color = self.COLORS['purple']
        self.is_recording = False
        self.show_help = False
        
        # # Smoothing buffers
        # self.point_buffer_x = deque(maxlen=self.smoothing_window)
        # self.point_buffer_y = deque(maxlen=self.smoothing_window)
        # Kalman filter for smooth finger tracking
        self.kalman = KalmanFilter2D(
            process_noise=0.1,   # lower = smoother but slightly more lag
            measure_noise=1.0     # higher = ignores jitter more
        )
        # Keep a small deque for interpolation gap-filling
        self.point_buffer_x = deque(maxlen=3)
        self.point_buffer_y = deque(maxlen=3)
        
        # Drawing state
        self.img_canvas = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
        self.xp, self.yp = 0, 0
        
        # Recording
        self.video_writer = None
        self.recording_start_time = None
        
        # Gesture cooldown
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0
        
        # Setup camera
        print("📷 Initializing camera...")
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {camera_index}")
            print("🔧 Trying alternative camera indices...")
            
            for alt_index in [1, 2]:
                self.cap = cv2.VideoCapture(alt_index)
                if self.cap.isOpened():
                    print(f"✓ Camera opened at index {alt_index}")
                    break
            
            if not self.cap.isOpened():
                raise RuntimeError("No camera found. Please check your webcam connection.")
        
        self.cap.set(3, self.WIDTH)
        self.cap.set(4, self.HEIGHT)
        print("✓ Camera initialized")
        
        # Hand detector
        print("👋 Initializing hand detector...")
        # self.detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.5)
        self.detector = HandDetector(max_hands=1, detection_con=0.5, track_con=0.7)
        
        # Create output directories
        os.makedirs('saved_drawings', exist_ok=True)
        os.makedirs('recordings', exist_ok=True)
        os.makedirs('extracted_text', exist_ok=True)
        print("✓ Output directories ready")
        
        print("\n" + "="*50)
        print("🎨 VIRTUAL PAINTER READY!")
        print("="*50)
        print("Press 'H' for help\n")
    
    # def get_smoothed_point(self, x, y):
    #     self.point_buffer_x.append(x)
    #     self.point_buffer_y.append(y)
        
    #     if len(self.point_buffer_x) > 0:
    #         smooth_x = int(np.mean(self.point_buffer_x))
    #         smooth_y = int(np.mean(self.point_buffer_y))
    #         return smooth_x, smooth_y
    #     return x, y
    
    # def clear_smoothing_buffer(self):
    #     """Clear smoothing buffer"""
    #     self.point_buffer_x.clear()
    #     self.point_buffer_y.clear()

    def get_smoothed_point(self, x, y):
        """
        Kalman-filter the raw fingertip position.
        Much better than plain moving average — handles fast movement
        without gaps or lag by using velocity prediction.
        """
        smooth_x, smooth_y = self.kalman.update(x, y)
        self.point_buffer_x.append(smooth_x)
        self.point_buffer_y.append(smooth_y)
        return smooth_x, smooth_y
 
    def clear_smoothing_buffer(self):
        """
        Called when finger lifts or mode changes.
        Resets Kalman velocity so next stroke starts clean.
        """
        self.kalman.reset()
        self.point_buffer_x.clear()
        self.point_buffer_y.clear()
 
    def interpolate_and_draw(self, canvas, x_start, y_start, x_end, y_end, color, thickness):
        """
        Draws the line between two points AND fills any gap if the finger
        moved too far in one frame (fast movement).
 
        Without this, fast strokes look like dashes. With this, they're solid.
        """
        dist = np.hypot(x_end - x_start, y_end - y_start)
 
        # If gap is large, insert intermediate points
        if dist > 20:
            steps = int(dist // 4)  # one fill point every 4px
            for i in range(1, steps):
                t = i / steps
                xi = int(x_start + t * (x_end - x_start))
                yi = int(y_start + t * (y_end - y_start))
                cv2.circle(canvas, (xi, yi), thickness // 2, color, -1)
 
        cv2.line(canvas, (x_start, y_start), (x_end, y_end), color, thickness)
    
    def draw_ui(self, img):
        """Draw modern UI overlay"""
        overlay = img.copy()
        
        # Top bar background
        cv2.rectangle(overlay, (0, 0), (self.WIDTH, self.HEADER_H), (40, 40, 40), -1)
        
        # Color palette
        colors_info = [
            ('Purple', self.COLORS['purple'], 50),
            ('Blue', self.COLORS['blue'], 180),
            ('Green', self.COLORS['green'], 310),
            ('Red', self.COLORS['red'], 440),
            ('Yellow', self.COLORS['yellow'], 570),
            ('Eraser', self.COLORS['black'], 700)
        ]
        
        for name, color, x in colors_info:
            is_active = (color == self.current_color)
            thickness = -1 if is_active else 3
            cv2.circle(overlay, (x, 50), 25, color, thickness)
            
            if is_active:
                cv2.circle(overlay, (x, 50), 32, (255, 255, 255), 2)
            
            cv2.putText(overlay, name, (x - 30, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Brush size indicator
        tools_x = 850
        cv2.putText(overlay, f"Size: {self.brush_thickness}", (tools_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.rectangle(overlay, (tools_x, 45), (tools_x + 100, 65), (60, 60, 60), -1)
        size_bar_width = min(100, int(self.brush_thickness * 2))
        cv2.rectangle(overlay, (tools_x, 45), (tools_x + size_bar_width, 65),
                     self.current_color, -1)
        
        # Recording indicator
        if self.is_recording:
            cv2.circle(overlay, (tools_x + 150, 50), 15, (0, 0, 255), -1)
            cv2.putText(overlay, "REC", (tools_x + 175, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Save button
        cv2.rectangle(overlay, (1050, 30), (1150, 70), (0, 200, 0), -1)
        cv2.putText(overlay, "SAVE", (1065, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Help button
        cv2.circle(overlay, (1200, 50), 20, (100, 100, 255), -1)
        cv2.putText(overlay, "?", (1193, 58),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        if self.show_help:
            self.draw_help_overlay(img)
        
        return img
    
    def draw_help_overlay(self, img):
        """Draw help instructions"""
        overlay = img.copy()
        cv2.rectangle(overlay, (200, 150), (1080, 570), (30, 30, 30), -1)
        cv2.rectangle(overlay, (200, 150), (1080, 570), (100, 200, 255), 3)
        
        help_text = [
            "GESTURE CONTROLS:",
            "",
            "Two fingers (Index + Middle) - Selection Mode",
            "Index finger only - Drawing Mode",
            "All fingers up - Clear Canvas",
            "Thumbs up - Increase brush size",
            "Index only (no thumb) - Decrease brush size",
            "Fist (all fingers down) - Pause drawing",
            "",
            "KEYBOARD SHORTCUTS:",
            "",
            "S - Save current drawing",
            "R - Start/Stop recording",
            "T - Extract text (OCR)" + (" [Available]" if OCR_AVAILABLE else " [Install pytesseract]"),
            "C - Clear canvas",
            "H - Toggle this help",
            "+ / - - Adjust brush size",
            "ESC - Exit application",
            "",
            "Press H or click ? to close this help"
        ]
        
        y_offset = 170
        for i, line in enumerate(help_text):
            if line.isupper() and line.endswith(':'):
                color = (100, 200, 255)
                font_scale = 0.7
            else:
                color = (255, 255, 255)
                font_scale = 0.5
            
            cv2.putText(overlay, line, (230, y_offset + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        cv2.addWeighted(overlay, 0.95, img, 0.05, 0, img)
    
    def process_gestures(self, fingers):
        """Handle gesture-based controls"""
        current_time = time.time()
        
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
        
        # Thumbs up - increase brush size
        if fingers == [1, 0, 0, 0, 0]:
            self.brush_thickness = min(50, self.brush_thickness + 5)
            self.last_gesture_time = current_time
            print(f"📏 Brush size: {self.brush_thickness}")
        
        # Index only (no thumb) - decrease brush size
        elif fingers == [0, 1, 0, 0, 0]:
            self.brush_thickness = max(5, self.brush_thickness - 5)
            self.last_gesture_time = current_time
            print(f"📏 Brush size: {self.brush_thickness}")
    
    def save_drawing(self):
        """Save current canvas"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"saved_drawings/drawing_{timestamp}.png"
            cv2.imwrite(filename, self.img_canvas)
            print(f"✅ Drawing saved: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    def toggle_recording(self):
        """Start or stop recording"""
        try:
            if not self.is_recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recordings/session_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(
                    filename, fourcc, 20.0, (self.WIDTH, self.HEIGHT)
                )
                self.is_recording = True
                self.recording_start_time = time.time()
                print(f"🔴 Recording started: {filename}")
            else:
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                self.is_recording = False
                duration = time.time() - self.recording_start_time
                print(f"⏹️  Recording stopped (Duration: {duration:.1f}s)")
        except Exception as e:
            print(f"❌ Recording error: {e}")
            self.is_recording = False
    
    # def extract_text_from_canvas(self):
    #     """Extract text using OCR"""
    #     if not OCR_AVAILABLE:
    #         print("❌ OCR not available")
    #         print("📦 Install: pip install pytesseract pillow")
    #         print("📦 Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
    #         return
        
    #     try:
    #         gray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)
    #         _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    #         pil_image = Image.fromarray(thresh)

    #         text = pytesseract.image_to_string(
    #             pil_image,
    #             config="--psm 6"
    #         )
            
    #         if text.strip():
    #             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #             filename = f"extracted_text/text_{timestamp}.txt"
    #             with open(filename, 'w', encoding='utf-8') as f:
    #                 f.write(text)
    #             print(f"📝 Text extracted: {filename}")
    #             print(f"Text preview:\n{text[:200]}...")
    #         else:
    #             print("⚠️  No text detected")
    #     except Exception as e:
    #         print(f"❌ OCR error: {e}")
    # ─────────────────────────────────────────────────────────────────────────────
# PASTE THIS into app.py — replaces the extract_text_from_canvas method
# inside the VirtualPainter class (around line 540)
# ─────────────────────────────────────────────────────────────────────────────

    def extract_text_from_canvas(self):
        """Extract text using OCR — fixed for black-background canvas"""
        if not OCR_AVAILABLE:
            print("❌ OCR not available")
            print("📦 Install: pip install pytesseract pillow")
            return

        try:
            # ── Step 1: Grayscale ──────────────────────────────────────────
            gray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)

            # ── Step 2: Upscale 2x (Tesseract needs bigger images) ─────────
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # ── Step 3: Smooth jagged air-drawn edges ──────────────────────
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # ── Step 4: INVERT threshold ───────────────────────────────────
            # Canvas = black background + colored strokes
            # Tesseract needs = white background + black text
            # THRESH_BINARY_INV + THRESH_OTSU does this automatically
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # ── Step 5: Dilate — thickens thin strokes ────────────────────
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thresh = cv2.dilate(thresh, kernel, iterations=1)

            # ── Step 6: Add padding — Tesseract fails on border text ───────
            padded = cv2.copyMakeBorder(
                thresh, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255
            )

            pil_image = Image.fromarray(padded)

            # ── Step 7: Try multiple PSM modes, pick best result ──────────
            best_text = ""
            for psm in [11, 6, 3, 7, 8]:
                config = f"--psm {psm} --oem 1"
                result = pytesseract.image_to_string(pil_image, config=config)
                cleaned = result.strip()
                if cleaned and len(cleaned) > len(best_text):
                    best_text = cleaned

            # ── Step 8: Save or report ─────────────────────────────────────
            if best_text:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"extracted_text/text_{timestamp}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(best_text)
                print(f"📝 Text extracted → {filename}")
                print(f"🔤 Detected:\n{'='*30}\n{best_text}\n{'='*30}")
            else:
                print("⚠️  No text detected.")
                print("💡 Tips: write LARGER letters, use bright color, increase brush size (+)")
                # Save debug image to inspect what Tesseract received
                debug_path = "extracted_text/debug_ocr_input.png"
                pil_image.save(debug_path)
                print(f"   Debug image → {debug_path}")

        except Exception as e:
            print(f"❌ OCR error: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Main application loop"""
        try:
            while True:
                success, img = self.cap.read()
                
                if not success:
                    print("❌ Failed to read camera frame")
                    break
                
                img = cv2.flip(img, 1)
                
                # Detect hand
                img = self.detector.find_hands(img, draw=False)
                lm_list, bbox = self.detector.find_position(img, draw=False)
                
                if len(lm_list) != 0:
                    x1, y1 = lm_list[8][1], lm_list[8][2]
                    x2, y2 = lm_list[12][1], lm_list[12][2]
                    
                    fingers = self.detector.fingers_up()
                    self.process_gestures(fingers)
                    
                    # Selection Mode
                    if fingers[1] == 1 and fingers[2] == 1:
                        self.xp, self.yp = 0, 0
                        self.clear_smoothing_buffer()
                        
                        if y1 < self.HEADER_H:
                            if 25 < x1 < 75:
                                self.current_color = self.COLORS['purple']
                            elif 155 < x1 < 205:
                                self.current_color = self.COLORS['blue']
                            elif 285 < x1 < 335:
                                self.current_color = self.COLORS['green']
                            elif 415 < x1 < 465:
                                self.current_color = self.COLORS['red']
                            elif 545 < x1 < 595:
                                self.current_color = self.COLORS['yellow']
                            elif 675 < x1 < 725:
                                self.current_color = self.COLORS['black']
                            elif 1050 < x1 < 1150:
                                self.save_drawing()
                            elif 1180 < x1 < 1220:
                                self.show_help = not self.show_help
                        
                        cv2.rectangle(img, (x1, y1 - 15), (x2, y2 + 15),
                                    self.current_color, cv2.FILLED)
                    
                    # Drawing Mode
                    # elif fingers[1] == 1 and fingers[2] == 0:
                    #     smooth_x, smooth_y = self.get_smoothed_point(x1, y1)
                    #     cv2.circle(img, (smooth_x, smooth_y), 10,
                    #              self.current_color, cv2.FILLED)
                        
                    #     if self.xp == 0 and self.yp == 0:
                    #         self.xp, self.yp = smooth_x, smooth_y
                        
                    #     thickness = (self.eraser_thickness if self.current_color == self.COLORS['black']
                    #                else self.brush_thickness)
                        
                    #     cv2.line(img, (self.xp, self.yp), (smooth_x, smooth_y),
                    #            self.current_color, thickness)
                    #     cv2.line(self.img_canvas, (self.xp, self.yp), (smooth_x, smooth_y),
                    #            self.current_color, thickness)
                        
                    #     self.xp, self.yp = smooth_x, smooth_y

                    elif fingers[1] == 1 and fingers[2] == 0:
                        smooth_x, smooth_y = self.get_smoothed_point(x1, y1)
                        cv2.circle(img, (smooth_x, smooth_y), 10,
                                 self.current_color, cv2.FILLED)
 
                        if self.xp == 0 and self.yp == 0:
                            self.xp, self.yp = smooth_x, smooth_y
 
                        thickness = (self.eraser_thickness if self.current_color == self.COLORS['black']
                                   else self.brush_thickness)
 
                        # Draw on live camera feed
                        self.interpolate_and_draw(
                            img, self.xp, self.yp, smooth_x, smooth_y,
                            self.current_color, thickness
                        )
                        # Draw on persistent canvas
                        self.interpolate_and_draw(
                            self.img_canvas, self.xp, self.yp, smooth_x, smooth_y,
                            self.current_color, thickness
                        )
 
                        self.xp, self.yp = smooth_x, smooth_y
                    
                    # Clear Canvas
                    elif all(f == 1 for f in fingers):
                        self.img_canvas = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
                        self.xp, self.yp = 0, 0
                        self.clear_smoothing_buffer()
                    
                    else:
                        self.xp, self.yp = 0, 0
                        self.clear_smoothing_buffer()
                
                else:
                    self.xp, self.yp = 0, 0
                    self.clear_smoothing_buffer()
                
                # Merge canvas
                img_gray = cv2.cvtColor(self.img_canvas, cv2.COLOR_BGR2GRAY)
                _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
                img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
                img = cv2.bitwise_and(img, img_inv)
                img = cv2.bitwise_or(img, self.img_canvas)
                
                # Draw UI
                img = self.draw_ui(img)
                
                # Record
                if self.is_recording and self.video_writer:
                    self.video_writer.write(img)
                
                # Display
                cv2.imshow("Virtual Painter", img)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('s') or key == ord('S'):
                    self.save_drawing()
                elif key == ord('r') or key == ord('R'):
                    self.toggle_recording()
                elif key == ord('t') or key == ord('T'):
                    self.extract_text_from_canvas()
                elif key == ord('c') or key == ord('C'):
                    self.img_canvas = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
                    print("🗑️  Canvas cleared")
                elif key == ord('h') or key == ord('H'):
                    self.show_help = not self.show_help
                elif key == ord('+') or key == ord('='):
                    self.brush_thickness = min(50, self.brush_thickness + 2)
                    print(f"📏 Brush size: {self.brush_thickness}")
                elif key == ord('-') or key == ord('_'):
                    self.brush_thickness = max(5, self.brush_thickness - 2)
                    print(f"📏 Brush size: {self.brush_thickness}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Runtime error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        if self.is_recording and self.video_writer:
            self.video_writer.release()
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        print("👋 Application closed successfully")


if __name__ == "__main__":
    try:
        painter = VirtualPainter(camera_index=0)
        painter.run()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 If you see MediaPipe errors, try:")
        print("   pip install mediapipe==0.10.9")