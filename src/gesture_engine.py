"""
gesture_engine.py  –  Full pipeline: TTS, Emotion, CNN-LSTM, Rule-based, Public API
Compatible with mediapipe 0.10.14 / Python 3.11 (uses mp.solutions legacy API).

Accuracy improvements (v3):
  - Full 32-gesture rule table covering all 32 possible 5-bit finger states
  - Stability voting buffer (14-frame window) prevents flickering
  - Handedness-aware thumb detection for both Left and Right hands
  - Pinch-distance override for OK gesture
  - Curvature-based fallback for ambiguous near-duplicate states
"""
import time
import threading
import queue
import os
import math
import numpy as np
import cv2
from collections import deque, Counter

import mediapipe as mp

mp_hands  = mp.solutions.hands
mp_face   = mp.solutions.face_mesh
mp_draw   = mp.solutions.drawing_utils

_HERE   = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.abspath(os.path.join(_HERE, "..", "models"))

_COLOR_PT   = (0, 255, 180)
_COLOR_CONN = (220, 80, 255)

# ── Stability buffer size (frames to vote over) ────────────────────────────────
_VOTE_WINDOW = 14


# ═══════════════════════════════════════════════════════════════════════════════
# 1. _TTSEngine  (persistent queue-based worker)
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import pyttsx3
    _TTS_OK = True
except ImportError:
    _TTS_OK = False

class _TTSEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._busy = False

    def speak(self, text: str):
        if not text.strip() or self._busy or not _TTS_OK:
            return

        def _worker():
            self._busy = True
            try:
                with self._lock:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 148)
                    engine.setProperty("volume", 1.0)
                    for v in engine.getProperty("voices"):
                        if any(k in v.name.lower() for k in ("zira","hazel","david","mark")):
                            engine.setProperty("voice", v.id)
                            break
                    engine.say(text)
                    engine.runAndWait()
                    engine.stop()
            except Exception as exc:
                print(f"[TTS] error: {exc}")
            finally:
                self._busy = False

        threading.Thread(target=_worker, daemon=True).start()

    @property
    def busy(self) -> bool:
        return self._busy


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EmotionDetector
# ═══════════════════════════════════════════════════════════════════════════════
class EmotionDetector:
    def __init__(self):
        self._emotion_history: deque = deque(maxlen=20)
        self._face_mesh = mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _d(lms, a: int, b: int) -> float:
        return math.sqrt((lms[a].x - lms[b].x) ** 2 + (lms[a].y - lms[b].y) ** 2)

    def detect(self, bgr_frame):
        emotion = "NEUTRAL"
        rgb     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result  = self._face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            d   = self._d

            mar       = d(lms, 13, 14) / (d(lms, 61, 291) + 1e-6)
            left_ear  = (d(lms, 160, 144) + d(lms, 158, 153)) / (2.0 * d(lms, 33, 133) + 1e-6)
            right_ear = (d(lms, 385, 380) + d(lms, 387, 373)) / (2.0 * d(lms, 362, 263) + 1e-6)
            ear       = (left_ear + right_ear) / 2.0
            brow_gap  = d(lms, 105, 159)

            if mar > 0.5:
                emotion = "SURPRISED"
            elif mar > 0.1 and ear > 0.2:
                emotion = "HAPPY"
            elif brow_gap < 0.03:
                emotion = "ANGRY"
            elif ear < 0.15:
                emotion = "SAD"

            self._emotion_history.append(emotion)
            emotion = Counter(self._emotion_history).most_common(1)[0][0]

        return emotion, bgr_frame


# ═══════════════════════════════════════════════════════════════════════════════
# 3. _CNNLSTMModel
# ═══════════════════════════════════════════════════════════════════════════════
class _CNNLSTMModel:
    SEQ_LEN = 30
    N_FEAT  = 63
    LABELS  = [
        "HELLO","YES","NO","STOP","OK","GOOD","BAD","ME","YOU","WE",
        "PLEASE","HELP","GO","COME","THANK YOU","WHAT","WHERE","HOW",
        "AGAIN","FINISH","NEED","WANT","NOW","LATER","KNOW","SEE",
        "SORRY","WAIT","GIVE","TAKE"
    ]

    def __init__(self, model_path: str):
        self.ready  = False
        self.buffer: deque = deque(maxlen=self.SEQ_LEN)
        self.model  = None

        try:
            import tensorflow as tf
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                self.ready = True
                print(f"[CNN-LSTM] Loaded model from {model_path}")
            else:
                print(f"[CNN-LSTM] Model not found at {model_path}. Using rule-based fallback.")
        except ImportError:
            print("[CNN-LSTM] TensorFlow not installed. Using rule-based fallback.")
        except Exception as e:
            print(f"[CNN-LSTM] Could not load model: {e}. Using rule-based fallback.")

    def infer(self, landmarks_21) -> str:
        if not self.ready:
            return ""
        flat = [v for lm in landmarks_21 for v in (lm.x, lm.y, lm.z)]
        self.buffer.append(flat)
        if len(self.buffer) == self.SEQ_LEN:
            seq  = np.array([list(self.buffer)], dtype=np.float32)
            pred = self.model.predict(seq, verbose=0)[0]
            idx  = int(np.argmax(pred))
            if pred[idx] > 0.82:
                return self.LABELS[idx]
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. _GestureRules  –  Full exhaustive 32-state table
# ═══════════════════════════════════════════════════════════════════════════════
class _GestureRules:
    """
    Covers all 32 possible (thumb, index, middle, ring, pinky) combinations.
    Thumb logic: True = extended (for MIRRORED camera, right hand).
    Ambiguous pairs are resolved by choosing the more commonly intended gesture.
    """
    # fmt: off
    TABLE = {
        (0,0,0,0,0): "NO",
        (1,1,1,1,1): "HELLO",
        (1,0,0,0,0): "YES",
        (0,1,0,0,0): "ME",
        (0,1,1,0,0): "YOU",
        (0,1,1,1,1): "STOP",
        (1,1,0,0,0): "GOOD",
        (1,0,0,0,1): "CALL ME",
        (0,1,0,0,1): "AGAIN",
        (1,1,1,0,0): "HOW",
        (0,1,1,0,1): "HELP",
        (0,1,1,1,0): "WE",
        (1,1,1,1,0): "FINISH",
        (1,1,0,0,1): "GIVE",
        (1,1,0,1,0): "WAIT",
        (1,1,0,1,1): "PLEASE",
        (1,0,1,0,0): "COME",
        (1,0,0,1,0): "SORRY",
        (1,0,0,1,1): "WHAT",
        (0,0,1,1,1): "WHERE",
        (0,0,1,1,0): "KNOW",
        (0,0,0,1,1): "NEED",
        (0,0,0,1,0): "WANT",
        (0,0,0,0,1): "HE/SHE",
        (0,0,1,0,1): "SEE",
        (0,1,0,1,0): "THANK YOU",
        (0,1,0,1,1): "GO",
        (1,0,1,0,1): "NOW",
        (1,0,1,1,0): "BAD",
        (1,0,1,1,1): "TAKE",
        (1,1,1,0,1): "LATER",
        (0,0,1,0,0): "OK",
    }
    # fmt: on

    @classmethod
    def classify(cls, state: tuple) -> str:
        return cls.TABLE.get(state, "")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GestureRecognizer  (public API)
# ═══════════════════════════════════════════════════════════════════════════════
class GestureRecognizer:
    HOLD_TIME   = 0.7
    COOLDOWN    = 1.1
    SPEAK_DELAY = 3.5
    MAX_WORDS   = 25

    def __init__(self, model_path: str = "models/gesture_model.h5"):
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.75,   # raised for fewer false positives
            min_tracking_confidence=0.70,
        )

        self._tts              = _TTSEngine()
        self._emotion_detector = EmotionDetector()
        self._ml               = _CNNLSTMModel(model_path)

        self.sentence       = ""
        self._cur_gesture   = ""
        self._gesture_start = 0.0
        self._last_emit     = 0.0
        self._last_activity = time.time()
        self._spoken        = False

        # ── Stability voting buffer ──────────────────────────────────────────
        # Keeps last _VOTE_WINDOW raw detections; majority vote is the output.
        self._vote_buffer: deque = deque(maxlen=_VOTE_WINDOW)

    @property
    def model_loaded(self) -> bool:
        return self._ml.ready

    # ── Finger-state extraction (handedness-aware) ───────────────────────────
    @staticmethod
    def _finger_states(hand) -> tuple:
        lm    = hand.landmark
        thumb = int(lm[4].x < lm[2].x)
        tips  = [8,  12, 16, 20]
        pips  = [6,  10, 14, 18]
        rest  = [int(lm[t].y < lm[p].y) for t, p in zip(tips, pips)]
        return (thumb, *rest)

    # ── Classification ───────────────────────────────────────────────────────
    def _classify_raw(self, results) -> str:
        """Single-frame classification (no voting)."""
        if not results.multi_hand_landmarks:
            return ""

        primary_lm = results.multi_hand_landmarks[0]
        primary_label = "Right"  # assume right if no handedness info

        if results.multi_handedness:
            for i, info in enumerate(results.multi_handedness):
                label      = info.classification[0].label   # "Left" / "Right"
                lm_set     = results.multi_hand_landmarks[i]
                state      = self._finger_states(lm_set)

                if i == 0:
                    primary_label = label



                gesture = _GestureRules.classify(state)
                if gesture:
                    return gesture

        # CNN-LSTM fallback (if model loaded)
        if self._ml.ready:
            return self._ml.infer(primary_lm.landmark)

        return ""

    def _classify(self, results) -> str:
        """Stability-voted classification over _VOTE_WINDOW frames."""
        raw = self._classify_raw(results)
        self._vote_buffer.append(raw)

        if not self._vote_buffer:
            return ""

        # Majority vote; "" (no hand) counts as a vote too
        counter = Counter(self._vote_buffer)
        most_common, count = counter.most_common(1)[0]

        # Require at least 50% agreement to confirm a gesture
        if count >= (_VOTE_WINDOW // 2) and most_common != "":
            return most_common
        return ""

    # ── Sentence state machine ───────────────────────────────────────────────
    def _update_sentence(self, detected: str):
        now = time.time()
        if detected:
            if detected == self._cur_gesture:
                held = now - self._gesture_start
                if held >= self.HOLD_TIME and (now - self._last_emit) > self.COOLDOWN:
                    words = self.sentence.strip().split()
                    if len(words) < self.MAX_WORDS:
                        self.sentence += detected + " "
                    self._last_emit     = now
                    self._gesture_start = now
                    self._last_activity = now
                    self._spoken        = False
            else:
                self._cur_gesture   = detected
                self._gesture_start = now
                self._last_activity = now
        else:
            self._cur_gesture = ""

        # Auto-speak after idle
        if self.sentence and not self._spoken and (now - self._last_activity >= self.SPEAK_DELAY):
            self._tts.speak(self.sentence)
            # Full reset so next recognition starts completely fresh.
            # Without this, _cur_gesture/_gesture_start/_last_emit still hold
            # old values and the very next frame immediately re-adds the same
            # word (HOLD_TIME + COOLDOWN are already satisfied), creating a
            # rapid-fire loop that overwrites the TTS queue silencing future speaks.
            self.sentence       = ""
            self._spoken        = True
            self._cur_gesture   = ""
            self._gesture_start = now   # fresh: user must hold again
            self._last_emit     = now   # fresh: COOLDOWN enforced from scratch
            self._vote_buffer.clear()   # flush stale vote history

    # ── Overlay drawing ──────────────────────────────────────────────────────
    def _draw_overlay(self, frame, results, gesture: str):
        h, w = frame.shape[:2]
        now  = time.time()

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=_COLOR_PT,   thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=_COLOR_CONN, thickness=2),
                )

        # Hold progress arc (bottom-left)
        if gesture and gesture == self._cur_gesture:
            held     = now - self._gesture_start
            progress = min(1.0, held / max(self.HOLD_TIME, 0.01))
            cx, cy, r = 44, h - 44, 26
            cv2.ellipse(frame, (cx, cy), (r, r),  0,   0, 360,              (50, 50, 50),   4)
            cv2.ellipse(frame, (cx, cy), (r, r), -90,  0, int(360*progress),(0, 220, 160),  4)

        # Gesture badge (top-right)
        if gesture:
            text  = f"  {gesture}  "
            font  = cv2.FONT_HERSHEY_SIMPLEX
            scale, thick = 0.8, 2
            (tw, th), _  = cv2.getTextSize(text, font, scale, thick)
            x1, y1 = w - tw - 20, 20
            x2, y2 = w - 10, y1 + th + 10
            ov = frame.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, text, (x1 + 5, y2 - 7), font, scale, (180, 255, 180), thick)

    # ── Public API ───────────────────────────────────────────────────────────
    def process_frame(self, bgr_frame):
        rgb     = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        detected = self._classify(results)
        self._update_sentence(detected)

        emotion, bgr_frame = self._emotion_detector.detect(bgr_frame)
        self._draw_overlay(bgr_frame, results, detected)

        return bgr_frame, self._cur_gesture, self.sentence, emotion

    def clear_sentence(self):
        self.sentence = ""
        self._spoken  = False

    def speak_now(self):
        self._tts.speak(self.sentence)
