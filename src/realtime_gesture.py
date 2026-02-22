import cv2
import mediapipe as mp
import time
import pyttsx3
import threading

class GestureRecognizer:
    def __init__(self):
        # ---------------- SETUP ----------------
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # ---------------- TEXT TO SPEECH ----------------
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 150)
        
        # ---------------- STATE ----------------
        self.sentence = ""
        self.current_gesture = ""
        self.gesture_start_time = 0
        self.last_emit_time = 0
        self.last_gesture_time = time.time()
        self.spoken = False
        
        self.HOLD_TIME = 0.8
        self.COOLDOWN = 1.2
        self.SPEAK_DELAY = 3.0

    # ---------------- UTIL FUNCTIONS ----------------
    def _finger_states(self, hand):
        lm = hand.landmark
        fingers = []

        # Thumb
        fingers.append(lm[4].x > lm[3].x)

        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]

        for tip, pip in zip(tips, pips):
            fingers.append(lm[tip].y < lm[pip].y)

        return tuple(fingers)  # (thumb, index, middle, ring, pinky)

    # ---------------- SINGLE HAND GESTURES ----------------
    def _detect_single_hand(self, f):
        if f == (1,1,1,1,1): return "HELLO"
        if f == (1,0,0,0,0): return "YES"
        if f == (0,0,0,0,1): return "NO"     # thumbs down style
        if f == (1,1,1,1,1): return "STOP"
        if f == (1,1,0,0,0): return "OK"
        if f == (1,0,0,0,0): return "GOOD"
        if f == (0,0,0,0,1): return "BAD"
        return ""

    # ---------------- TWO HAND GESTURES ----------------
    def _detect_two_hand(self, left, right):
        if left == (0,1,0,0,0) and right == (0,1,0,0,0): return "YOU"
        if left == (1,0,0,0,0) and right == (0,0,0,0,0): return "ME"
        if left == (0,1,1,0,0) and right == (0,1,1,0,0): return "WE"
        if left == (1,1,1,1,1) and right == (1,1,1,1,1): return "PLEASE"
        if left == (1,0,0,0,0) and right == (1,1,1,1,1): return "HELP"
        if left == (0,1,0,0,0) and right == (1,0,0,0,0): return "GO"
        if left == (1,0,0,0,0) and right == (0,1,0,0,0): return "COME"
        if left == (1,0,0,0,0) and right == (1,1,1,1,1): return "THANKYOU"
        if left == (0,1,1,0,0) and right == (0,0,0,0,0): return "WHAT"
        if left == (0,0,0,0,0) and right == (0,1,1,0,0): return "WHERE"
        if left == (0,1,0,0,1) and right == (0,1,0,0,1): return "HOW"
        if left == (0,0,0,0,0) and right == (0,0,0,0,0): return "AGAIN"
        if left == (1,1,1,1,1) and right == (0,0,0,0,0): return "FINISH"
        return ""

    def speak(self, text):
        """Runs TTS in a separate thread to prevent UI freezing"""
        def run():
            self.engine.say(text)
            self.engine.runAndWait()
        threading.Thread(target=run, daemon=True).start()

    def process_frame(self, frame):
        """
        Processes a single frame:
        1. Detects hands & landmarks
        2. Recognizes gestures
        3. Updates state (sentence building)
        4. Returns processed frame (RGB) and current state info
        """
        
        # Convert for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        left = None
        right = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, info in enumerate(results.multi_handedness):
                label = info.classification[0].label
                lm = results.multi_hand_landmarks[idx]
                
                # Draw landmarks on the frame (modify in place or copy if strict)
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
                
                if label == "Left":
                    left = self._finger_states(lm)
                else:
                    right = self._finger_states(lm)

        detected = ""
        
        if left and right:
            detected = self._detect_two_hand(left, right)
            self.last_gesture_time = time.time()
        elif left or right:
            detected = self._detect_single_hand(left or right)
            self.last_gesture_time = time.time()
            
        if detected:
            if detected != self.current_gesture:
                self.current_gesture = detected
                self.gesture_start_time = time.time()
            elif time.time() - self.gesture_start_time > self.HOLD_TIME:
                if time.time() - self.last_emit_time > self.COOLDOWN:
                    self.sentence += detected + " "
                    self.last_emit_time = time.time()
                    self.gesture_start_time = time.time()
                    self.spoken = False
        else:
            self.current_gesture = ""

        # -------- AUTO SPEAK --------
        if self.sentence and not self.spoken and (time.time() - self.last_gesture_time > self.SPEAK_DELAY):
            self.speak(self.sentence)
            self.sentence = "" # Reset after speaking? Or keep? Resetting seems logical for single sentence.
            self.spoken = True
            self.last_gesture_time = time.time()

        return frame, self.current_gesture, self.sentence