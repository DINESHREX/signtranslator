import cv2
import mediapipe as mp
import pyttsx3
import time
import math

# ---------------- TTS ----------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

# ---------------- STATE ----------------
sentence = ""
last_gesture = ""
last_time = time.time()

GESTURE_DELAY = 1.2
SPEAK_DELAY = 3.0

# ---------------- HELPERS ----------------
def finger_up(tip, pip):
    return tip.y < pip.y

def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

print("🚀 Rule-Based Sign Translator (30 Gestures) | Press Q to quit")

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = ""
    now = time.time()

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark

        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,255), thickness=2),
            mp_draw.DrawingSpec(color=(255,0,255), thickness=2)
        )

        thumb = lm[4].x < lm[3].x
        index = finger_up(lm[8], lm[6])
        middle = finger_up(lm[12], lm[10])
        ring = finger_up(lm[16], lm[14])
        pinky = finger_up(lm[20], lm[18])

        fingers = [index, middle, ring, pinky]
        count = fingers.count(True)

        # ---------- RULES (30 gestures) ----------
        if count == 5:
            gesture = "HELLO"

        elif thumb and count == 0:
            gesture = "YES"

        elif not thumb and count == 0:
            gesture = "NO"

        elif count == 4 and not thumb:
            gesture = "STOP"

        elif dist(lm[4], lm[8]) < 0.05:
            gesture = "GOOD"

        elif count == 1 and index:
            gesture = "ME"

        elif count == 2 and index and middle:
            gesture = "YOU"

        elif count == 3:
            gesture = "WE"

        elif count == 4 and thumb:
            gesture = "PLEASE"

        elif count == 0 and thumb:
            gesture = "THANK YOU"

        elif index and lm[8].x > lm[6].x:
            gesture = "GO"

        elif index and lm[8].x < lm[6].x:
            gesture = "COME"

        elif index and middle and ring and not pinky:
            gesture = "GIVE"

        elif index and not middle and not ring and not pinky and lm[8].y > lm[6].y:
            gesture = "TAKE"

        elif count == 1 and not thumb:
            gesture = "NEED"

        elif count == 2 and not thumb:
            gesture = "WANT"

        elif index and middle and not ring:
            gesture = "SEE"

        elif index and middle and ring:
            gesture = "KNOW"

        elif count == 1 and thumb:
            gesture = "NOW"

        elif count == 2 and thumb:
            gesture = "LATER"

        elif count == 5 and thumb:
            gesture = "FINISH"

        elif count == 3 and thumb:
            gesture = "AGAIN"

        elif count == 4:
            gesture = "WAIT"

        elif count == 1 and pinky:
            gesture = "HE"

        elif count == 1 and ring:
            gesture = "SHE"

        elif count == 2 and ring and pinky:
            gesture = "THEY"

        elif count == 0:
            gesture = "BAD"

        elif thumb and index:
            gesture = "OK"

        elif index and middle and pinky:
            gesture = "HELP"

        elif count == 3 and not thumb:
            gesture = "SORRY"

    # ---------- SENTENCE ----------
    if gesture and gesture != last_gesture and now - last_time > GESTURE_DELAY:
        sentence += gesture + " "
        last_gesture = gesture
        last_time = now

    # ---------- SPEAK ----------
    if sentence and now - last_time > SPEAK_DELAY:
        engine.say(sentence)
        engine.runAndWait()
        sentence = ""

    # ---------- UI ----------
    cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.putText(frame, f"Sentence: {sentence}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180,255,180), 2)

    cv2.imshow("Sign Translator (30 Gestures)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()