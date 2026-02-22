"""
collect_data.py  –  Collects CNN-LSTM training sequences.
Compatible with mediapipe 0.10.14 / Python 3.11 (uses mp.solutions legacy API).
"""
import cv2
import numpy as np
import os
import time

import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# ─────────────────────────── CONFIG ────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SEQ_LEN     = 30
TARGET_SEQS = 40
N_FEAT      = 63

GESTURES = [
    "HELLO","YES","NO","STOP","OK","GOOD","BAD","ME","YOU","WE",
    "PLEASE","HELP","GO","COME","THANK YOU","WHAT","WHERE","HOW",
    "AGAIN","FINISH","NEED","WANT","NOW","LATER","KNOW","SEE",
    "SORRY","WAIT","GIVE","TAKE"
]

os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────── MEDIAPIPE ──────────────────────────
_hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)


# ─────────────────────────── HELPERS ────────────────────────────
def _landmarks_to_vec(results) -> list:
    """Return flat [x, y, z, …] (63 floats) or all-zeros if no hand."""
    if results.multi_hand_landmarks:
        lms = results.multi_hand_landmarks[0].landmark
        return [v for lm in lms for v in (lm.x, lm.y, lm.z)]
    return [0.0] * N_FEAT


def _overlay(frame, line1: str, line2: str = "", line3: str = "", recording: bool = False):
    """Draw semi-transparent header bar with status text."""
    h, w      = frame.shape[:2]
    bar_color = (0, 0, 180) if recording else (20, 20, 20)
    overlay   = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), bar_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, line1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 160), 2)
    if line2:
        cv2.putText(frame, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    if line3:
        cv2.putText(frame, line3, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)


def _detect_and_draw(frame):
    """Run hands on a BGR frame, draw landmarks, return (results, frame)."""
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands.process(rgb)
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 180), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(220, 80, 255), thickness=2),
            )
    return results, frame


# ─────────────────────────── COLLECTION ─────────────────────────
def collect_gesture(gesture: str, already: int, cap) -> tuple:
    """
    Interactively collect sequences for *gesture*.
    Returns (new_seqs: list[np.ndarray], quit_flag: bool).
    """
    new_seqs = []
    done     = already

    while done < TARGET_SEQS:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        _, frame = _detect_and_draw(frame)

        remaining = TARGET_SEQS - done
        _overlay(frame,
                 f"Gesture: {gesture}",
                 f"Sequences: {done}/{TARGET_SEQS}   (need {remaining} more)",
                 "S=Record  R=Undo  Q=Next gesture  ESC=Quit all")

        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── 's': record one sequence ──
        if key == ord('s'):
            sequence = []
            for _ in range(SEQ_LEN):
                ret, frame = cap.read()
                if not ret:
                    break
                frame         = cv2.flip(frame, 1)
                results, frame = _detect_and_draw(frame)
                vec            = _landmarks_to_vec(results)
                sequence.append(vec)

                _overlay(frame,
                         f"RECORDING: {gesture}",
                         f"Frame {len(sequence)}/{SEQ_LEN}",
                         recording=True)
                cv2.imshow("Data Collector", frame)
                cv2.waitKey(1)

            if len(sequence) == SEQ_LEN:
                new_seqs.append(np.array(sequence, dtype=np.float32))
                done += 1
                print(f"  [{gesture}] Sequence {done}/{TARGET_SEQS} saved.")

        # ── 'r': undo last sequence ──
        elif key == ord('r'):
            if new_seqs:
                new_seqs.pop()
                done -= 1
                print(f"  [{gesture}] Last sequence removed. Total now: {done}")

        # ── 'q': move to next gesture ──
        elif key == ord('q'):
            break

        # ── ESC: quit everything ──
        elif key == 27:
            return new_seqs, True

    return new_seqs, False


# ─────────────────────────── MAIN ───────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    for gesture in GESTURES:
        npy_path = os.path.join(DATA_DIR, f"{gesture}.npy")

        existing = None
        if os.path.exists(npy_path):
            try:
                existing = np.load(npy_path)
                print(f"[{gesture}] Loaded {len(existing)} existing sequences.")
            except Exception:
                existing = None

        already = len(existing) if existing is not None else 0

        if already >= TARGET_SEQS:
            print(f"[{gesture}] Already has {already} sequences — skipping.")
            continue

        # 3-second countdown
        for countdown in range(3, 0, -1):
            start = time.time()
            while time.time() - start < 1.0:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                _overlay(frame, f"Next: {gesture}", f"Starting in {countdown}…", "Get ready!")
                cv2.imshow("Data Collector", frame)
                cv2.waitKey(1)

        new_seqs, quit_flag = collect_gesture(gesture, already, cap)

        # Save / merge
        if new_seqs:
            new_arr = np.array(new_seqs, dtype=np.float32)
            merged  = np.concatenate([existing, new_arr], axis=0) if existing is not None else new_arr
            np.save(npy_path, merged)
            print(f"[{gesture}] Saved {len(merged)} total sequences → {npy_path}")

        if quit_flag:
            print("ESC pressed — stopping early.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!  Next step: python src/train_model.py")


if __name__ == "__main__":
    main()
