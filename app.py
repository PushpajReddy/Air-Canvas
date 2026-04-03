import cv2
import numpy as np
import time
import os
from datetime import datetime
from collections import deque
import HandTrackingModule as htm
 
# Try importing OCR libraries
try:
    from PIL import Image
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:/Program Files/Tesseract-OCR/tesseract.exe"
    )
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: PIL/pytesseract not available. Text recognition disabled.")
 
class VirtualPainter:
    def __init__(self, camera_index=0):
        # Canvas settings
        self.WIDTH = 1280
        self.HEIGHT = 720
        self.HEADER_H = 130
 
        # Drawing parameters
        self.brushThickness = 15
        self.eraserThickness = 80
 
        # Colors
        self.COLORS = {
            'purple': (255, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'black': (0, 0, 0)
        }
 
        # UI State
        self.current_color = self.COLORS['purple']
        self.current_tool = 'brush'
        self.is_recording = False
        self.show_help = False
 
        # ── Mode debouncing ────────────────────────────────────────────────
        # Require the same gesture N frames in a row before switching modes.
        # Prevents a single bad MediaPipe frame from interrupting a stroke.
        self._mode_buffer = deque(maxlen=4)   # last 4 finger readings
        self._current_mode = 'idle'           # 'draw', 'select', 'idle'
 
        # Canvas and drawing state
        self.imgCanvas = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
        self.xp, self.yp = 0, 0
 
        # Recording
        self.video_writer = None
        self.recording_start_time = None
 
        # Gesture cooldown (for brush-size gestures, not drawing)
        self.last_gesture_time = 0
        self.gesture_cooldown = 1.0
 
        # Setup camera
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(3, self.WIDTH)
        self.cap.set(4, self.HEIGHT)
 
        # Hand detector — EMA smoothing + axis-relative fingersUp live here
        self.detector = htm.handDetector(detectionCon=0.5, trackCon=0.7, maxHands=1)
 
        # Create output directories
        os.makedirs('saved_drawings', exist_ok=True)
        os.makedirs('recordings', exist_ok=True)
        os.makedirs('extracted_text', exist_ok=True)
 
    # ── Gesture → mode (debounced) ─────────────────────────────────────────
    def _fingers_to_raw_mode(self, fingers):
        """Map finger state to a raw mode string."""
        if fingers[1] == 1 and fingers[2] == 0:
            return 'draw'
        elif fingers[1] == 1 and fingers[2] == 1:
            return 'select'
        elif all(f == 1 for f in fingers):
            return 'clear'
        return 'idle'
 
    def get_stable_mode(self, fingers):
        """
        Only switch mode when the same reading appears ≥3 times in the
        last 4 frames. This absorbs single-frame glitches from MediaPipe
        that would otherwise break a stroke mid-draw.
        """
        raw = self._fingers_to_raw_mode(fingers)
        self._mode_buffer.append(raw)
 
        # Majority vote over buffer
        counts = {}
        for m in self._mode_buffer:
            counts[m] = counts.get(m, 0) + 1
        dominant = max(counts, key=counts.get)
 
        # Only switch if dominant mode has ≥3 votes
        if counts[dominant] >= 3:
            self._current_mode = dominant
        return self._current_mode
 
    # ── Stroke gap filling ─────────────────────────────────────────────────
    def fill_stroke_gap(self, canvas, x0, y0, x1, y1, color, thickness):
        """
        Fill gaps caused by fast finger movement.
        Inserts intermediate circles every 4px so fast strokes stay solid
        instead of looking dashed.
        """
        dist = int(np.hypot(x1 - x0, y1 - y0))
        if dist > 20:
            steps = max(1, dist // 4)
            for i in range(1, steps):
                t = i / steps
                xi = int(x0 + t * (x1 - x0))
                yi = int(y0 + t * (y1 - y0))
                cv2.circle(canvas, (xi, yi), max(1, thickness // 2), color, -1)
        cv2.line(canvas, (x0, y0), (x1, y1), color, thickness)
 
    def clear_smoothing_buffer(self):
        """Reset drawing state when pen lifts."""
        self.xp, self.yp = 0, 0
        self._mode_buffer.clear()
    
    def draw_ui(self, img):
        """Draw modern UI overlay"""
        # Semi-transparent overlay for controls
        overlay = img.copy()
        
        # Top bar
        cv2.rectangle(overlay, (0, 0), (self.WIDTH, 100), (40, 40, 40), -1)
        
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
            # Draw color circle
            is_active = (color == self.current_color)
            thickness = -1 if is_active else 3
            cv2.circle(overlay, (x, 50), 25, color, thickness)
            if is_active:
                cv2.circle(overlay, (x, 50), 32, (255, 255, 255), 2)
            
            # Label
            cv2.putText(overlay, name, (x-30, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Tools section
        tools_x = 850
        
        # Brush size indicator
        cv2.putText(overlay, f"Size: {self.brushThickness}", (tools_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.rectangle(overlay, (tools_x, 45), (tools_x + 100, 65), (60, 60, 60), -1)
        cv2.rectangle(overlay, (tools_x, 45), 
                     (tools_x + int(self.brushThickness * 3), 65), 
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
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Help overlay
        if self.show_help:
            self.draw_help_overlay(img)
        
        return img
    
    def draw_help_overlay(self, img):
        """Draw help instructions"""
        overlay = img.copy()
        cv2.rectangle(overlay, (200, 150), (1080, 550), (30, 30, 30), -1)
        
        help_text = [
            "GESTURE CONTROLS:",
            "",
            "✌️  Two fingers (Index + Middle) - Selection Mode",
            "☝️  Index finger only - Drawing Mode",
            "🖐️ All fingers up - Clear Canvas",
            "👍 Thumbs up - Increase brush size",
            "👎 Thumbs down - Decrease brush size",
            "✊ Fist (all fingers down) - Pause drawing",
            "",
            "KEYBOARD SHORTCUTS:",
            "",
            "S - Save current drawing",
            "R - Start/Stop recording",
            "T - Extract text (OCR)",
            "C - Clear canvas",
            "H - Toggle this help",
            "+ / - Adjust brush size",
            "ESC - Exit application"
        ]
        
        y_offset = 180
        for i, line in enumerate(help_text):
            color = (100, 200, 255) if line.isupper() else (255, 255, 255)
            font_scale = 0.6 if line.isupper() else 0.5
            cv2.putText(overlay, line, (230, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
    
    def process_gestures(self, fingers):
        """Handle gesture-based controls with cooldown"""
        current_time = time.time()
        
        if current_time - self.last_gesture_time < self.gesture_cooldown:
            return
        
        # Thumbs up - increase brush size
        if fingers == [1, 0, 0, 0, 0]:
            self.brushThickness = min(50, self.brushThickness + 5)
            self.last_gesture_time = current_time
            print(f"Brush size increased to {self.brushThickness}")
        
        # Thumb down + index - decrease brush size
        elif fingers == [0, 1, 0, 0, 0]:
            self.brushThickness = max(5, self.brushThickness - 5)
            self.last_gesture_time = current_time
            print(f"Brush size decreased to {self.brushThickness}")
    
    def save_drawing(self):
        """Save current canvas to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_drawings/drawing_{timestamp}.png"
        cv2.imwrite(filename, self.imgCanvas)
        print(f"✅ Drawing saved: {filename}")
        return filename
    
    def toggle_recording(self, img):
        """Start or stop video recording"""
        if not self.is_recording:
            # Start recording
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
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.is_recording = False
            duration = time.time() - self.recording_start_time
            print(f"⏹️  Recording stopped. Duration: {duration:.1f}s")
    
    # def extract_text_from_canvas(self):
    #     """Extract handwritten text using OCR"""
    #     if not OCR_AVAILABLE:
    #         print("❌ OCR not available. Install: pip install pytesseract pillow")
    #         return
        
    #     # Convert canvas to PIL Image
    #     canvas_rgb = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2RGB)
    #     pil_image = Image.fromarray(canvas_rgb)
        
    #     # Perform OCR
    #     try:
    #         text = pytesseract.image_to_string(pil_image)
            
    #         if text.strip():
    #             # Save extracted text
    #             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #             filename = f"extracted_text/text_{timestamp}.txt"
    #             with open(filename, 'w') as f:
    #                 f.write(text)
    #             print(f"📝 Text extracted and saved: {filename}")
    #             print(f"Extracted text:\n{text}")
    #         else:
    #             print("⚠️  No text detected in drawing")
    #     except Exception as e:
    #         print(f"❌ OCR error: {e}")
    







    def extract_text_from_canvas(self):
        """Extract text using OCR — fixed for black-background canvas"""
        if not OCR_AVAILABLE:
            print("❌ OCR not available")
            print("📦 Install: pip install pytesseract pillow")
            return
 
        try:
            # ── Step 1: Grayscale ──────────────────────────────────────────
            gray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
 
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
                # config = f"--psm {psm} --oem 1"
                config = "--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
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
        print("🎨 Enhanced Virtual Painter Started!")
        print("Press 'H' for help")
        
        while True:
            success, img = self.cap.read()
            if not success:
                print("❌ Failed to read camera frame")
                break
            
            img = cv2.flip(img, 1)
            
            # Detect hand
            img = self.detector.findHands(img, draw=False)
            lmList, bbox = self.detector.findPosition(img, draw=False)
            
            if len(lmList) != 0:
                x1, y1 = lmList[8][1], lmList[8][2]   # Index fingertip
                x2, y2 = lmList[12][1], lmList[12][2]  # Middle fingertip
 
                fingers = self.detector.fingersUp()
                self.process_gestures(fingers)
 
                # Debounced mode — won't flicker on a single bad frame
                mode = self.get_stable_mode(fingers)
 
                # ── Selection mode ─────────────────────────────────────────
                if mode == 'select':
                    self.xp, self.yp = 0, 0
 
                    if y1 < 100:
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
 
                # ── Drawing mode ───────────────────────────────────────────
                elif mode == 'draw':
                    cv2.circle(img, (x1, y1), 10, self.current_color, cv2.FILLED)
 
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = x1, y1
 
                    thickness = self.eraserThickness if self.current_color == self.COLORS['black'] else self.brushThickness
                    # Gap-filling draw: solid strokes even when moving fast
                    self.fill_stroke_gap(img, self.xp, self.yp, x1, y1,
                                         self.current_color, thickness)
                    self.fill_stroke_gap(self.imgCanvas, self.xp, self.yp, x1, y1,
                                         self.current_color, thickness)
                    self.xp, self.yp = x1, y1
 
                # ── Clear canvas ───────────────────────────────────────────
                elif mode == 'clear':
                    self.imgCanvas = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
                    self.clear_smoothing_buffer()
 
                # ── Idle ───────────────────────────────────────────────────
                else:
                    self.xp, self.yp = 0, 0
 
            else:
                # Hand not visible — reset stroke anchor
                self.xp, self.yp = 0, 0
                self._mode_buffer.clear()
            
            # Merge canvas with image
            imgGray = cv2.cvtColor(self.imgCanvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, imgInv)
            img = cv2.bitwise_or(img, self.imgCanvas)
            
            # Draw UI
            img = self.draw_ui(img)
            
            # Record frame if recording
            if self.is_recording and self.video_writer:
                self.video_writer.write(img)
            
            # Display
            cv2.imshow("Enhanced Virtual Painter", img)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('s') or key == ord('S'):
                self.save_drawing()
            elif key == ord('r') or key == ord('R'):
                self.toggle_recording(img)
            elif key == ord('t') or key == ord('T'):
                self.extract_text_from_canvas()
            elif key == ord('c') or key == ord('C'):
                self.imgCanvas = np.zeros((self.HEIGHT, self.WIDTH, 3), np.uint8)
            elif key == ord('h') or key == ord('H'):
                self.show_help = not self.show_help
            elif key == ord('+') or key == ord('='):
                self.brushThickness = min(50, self.brushThickness + 2)
            elif key == ord('-') or key == ord('_'):
                self.brushThickness = max(5, self.brushThickness - 2)
        
        # Cleanup
        if self.is_recording and self.video_writer:
            self.video_writer.release()
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("👋 Application closed")
 
 
if __name__ == "__main__":
    painter = VirtualPainter(camera_index=0)
    painter.run()
