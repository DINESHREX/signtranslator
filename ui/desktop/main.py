import sys
import os
import time
import cv2

# ── Project root on path ──────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.gesture_engine import GestureRecognizer
from src.stt import SpeechListener

# ── Qt: PyQt5 with PySide6 fallback ──────────────────────────────────────────
try:
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
        QPushButton, QFrame, QSizePolicy, QSlider, QCheckBox,
        QStackedWidget
    )
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import QTimer, Qt
    _QT = "PyQt5"
except ImportError:
    from PySide6.QtWidgets import (
        QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
        QPushButton, QFrame, QSizePolicy, QSlider, QCheckBox,
        QStackedWidget
    )
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtCore import QTimer, Qt
    _QT = "PySide6"

# ── Emotion display maps ──────────────────────────────────────────────────────
_EMOTION_COLOR = {
    "HAPPY":     "#22c55e",
    "SAD":       "#60a5fa",
    "ANGRY":     "#ef4444",
    "SURPRISED": "#f59e0b",
    "NEUTRAL":   "#94a3b8",
}
_EMOTION_ICON = {
    "HAPPY":     "😊",
    "SAD":       "😢",
    "ANGRY":     "😠",
    "SURPRISED": "😲",
    "NEUTRAL":   "😐",
}


# ── Reusable helpers ──────────────────────────────────────────────────────────
def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setObjectName("separator")
    return line


def _make_card(title: str):
    """Returns (card_frame, value_label)."""
    card = QFrame()
    card.setObjectName("infoCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(18, 14, 18, 14)
    layout.setSpacing(6)

    lbl_title = QLabel(title)
    lbl_title.setObjectName("cardLabel")

    lbl_value = QLabel("—")
    lbl_value.setObjectName("cardValue")
    lbl_value.setAlignment(Qt.AlignCenter)

    layout.addWidget(lbl_title)
    layout.addWidget(lbl_value)
    return card, lbl_value

from PyQt5.QtWidgets import QTextEdit, QProgressBar, QStackedWidget, QSizePolicy, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import threading
import pyttsx3
class TextToSignWorker(QThread):
    """
    Runs the word-by-word playback in a background thread.
    Emits signals so the UI updates safely from the main thread.
    word_signal   — emits (word, hand_shape_description, index, total)
    done_signal   — emits when all words are finished
    """
    word_signal = pyqtSignal(str, str, int, int)
    done_signal = pyqtSignal()

    # Same gesture table from gesture_engine.py — maps word to hand shape description
    GESTURE_GUIDE = {
        "NO":        "Closed fist — all fingers curled down",
        "HELLO":     "Open palm — all 5 fingers extended up",
        "YES":       "Thumbs up — only thumb extended",
        "ME":        "Index finger only — point at yourself",
        "YOU":       "Peace / V sign — Index + Middle up",
        "STOP":      "4 fingers up — thumb tucked into palm",
        "GOOD":      "Thumb + Index up — L shape / gun shape",
        "CALL ME":   "Thumb + Pinky — phone sign / shaka",
        "AGAIN":     "Index + Pinky — rock horns sign",
        "HOW":       "Thumb + Index + Middle up",
        "HELP":      "Index + Middle + Pinky up",
        "WE":        "Index + Middle + Ring up",
        "FINISH":    "All fingers up except Pinky",
        "GIVE":      "Thumb + Index + Pinky up",
        "WAIT":      "Thumb + Index + Ring up",
        "PLEASE":    "Thumb + Index + Ring + Pinky up",
        "COME":      "Thumb + Middle only",
        "SORRY":     "Thumb + Ring only",
        "WHAT":      "Thumb + Ring + Pinky up",
        "WHERE":     "Middle + Ring + Pinky up",
        "KNOW":      "Middle + Ring only",
        "NEED":      "Ring + Pinky only",
        "WANT":      "Ring finger only",
        "HE/SHE":    "Pinky only — all others curled",
        "SEE":       "Middle + Pinky up (skip ring)",
        "THANK YOU": "Index + Ring up (skip middle)",
        "GO":        "Index + Ring + Pinky up",
        "NOW":       "Thumb + Middle + Pinky up",
        "BAD":       "Thumb + Middle + Ring up",
        "TAKE":      "Thumb + Middle + Ring + Pinky up",
        "LATER":     "Thumb + Index + Middle + Pinky up",
        "OK":        "Middle finger only",
    }

    def __init__(self, words: list, interval: float = 2.5):
        super().__init__()
        self.words    = words
        self.interval = interval
        self._paused  = False
        self._stopped = False

    def run(self):
        total = len(self.words)
        for i, word in enumerate(self.words):
            if self._stopped:
                break

            # Wait while paused
            while self._paused and not self._stopped:
                self.msleep(100)

            if self._stopped:
                break

            upper = word.upper()
            shape = self.GESTURE_GUIDE.get(upper, f"Spell it out: {word}")
            self.word_signal.emit(word, shape, i + 1, total)

            # Speak the word in a fresh pyttsx3 engine (fixes silent-after-first bug)
            def _speak(w):
                try:
                    eng = pyttsx3.init()
                    eng.setProperty("rate", 160)
                    eng.setProperty("volume", 1.0)
                    eng.say(w)
                    eng.runAndWait()
                    eng.stop()
                except Exception as e:
                    print(f"[TTS] {e}")
            threading.Thread(target=_speak, args=(word,), daemon=True).start()

            # Wait for interval, checking for pause/stop every 100ms
            elapsed = 0
            while elapsed < self.interval * 1000:
                if self._stopped:
                    break
                while self._paused and not self._stopped:
                    self.msleep(100)
                self.msleep(100)
                elapsed += 100

        if not self._stopped:
            self.done_signal.emit()

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop_playback(self):
        self._stopped = True
        self._paused  = False


# ── Main Window ──────────────────────────────────────────────────────────────
class SignTranslatorUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SignTalk — AI Sign Language Translator")
        self.setMinimumSize(1280, 720)
        self.setAttribute(Qt.WA_StyledBackground, True)

        # Camera
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(1)

        # Backend engines
        model_path = os.path.join(_ROOT, "models", "gesture_model.h5")
        self._rec = GestureRecognizer(model_path=model_path)
        self._stt = SpeechListener(callback=self._on_speech)

        # State
        self._fps_times  = []
        self._last_img   = None   # keeps QImage alive → prevents garbage-collection segfault
        self._emotion_on = True

        self._build_ui()
        self._load_styles()

        # Frame timer (≈30 fps)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._timer.start(33)

        self._tts_worker = None   # will hold TextToSignWorker instance

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_sidebar())

        self._stack = QStackedWidget()
        self._stack.addWidget(self._page_translator())   # 0
        self._stack.addWidget(self._build_tts_panel())   # 1
        self._stack.addWidget(self._page_settings())     # 2
        self._stack.addWidget(self._page_about())        # 3
        root.addWidget(self._stack, 1)
        # Apply the pending initial navigation now that _stack exists
        self._go(self._go_pending)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setFixedWidth(230)
        sidebar.setObjectName("sidebar")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 28, 16, 24)
        layout.setSpacing(8)

        logo = QLabel("🤟")
        logo.setObjectName("sidebarLogo")
        logo.setAlignment(Qt.AlignCenter)

        title = QLabel("SignTalk")
        title.setObjectName("sidebarTitle")
        title.setAlignment(Qt.AlignCenter)

        sub = QLabel("AI Sign Translator")
        sub.setObjectName("sidebarSub")
        sub.setAlignment(Qt.AlignCenter)

        layout.addWidget(logo)
        layout.addWidget(title)
        layout.addWidget(sub)
        layout.addSpacing(12)
        layout.addWidget(_separator())
        layout.addSpacing(6)

        # Nav buttons
        self._nav_btns = {}
        nav_items = [
            ("translator", "🎥  Live Translator"),
            ("tts",        "✍  Text to Sign"),
            ("settings",   "⚙️  Settings"),
            ("about",      "ℹ️  About"),
        ]
        for key, label in nav_items:
            btn = QPushButton(label)
            btn.setObjectName("sideButton")
            btn.setFixedHeight(44)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _, k=key: self._go(k))
            self._nav_btns[key] = btn
            layout.addWidget(btn)

        layout.addStretch()

        self._lbl_fps = QLabel("FPS: —")
        self._lbl_fps.setObjectName("statChip")
        self._lbl_fps.setAlignment(Qt.AlignCenter)

        self._lbl_mode = QLabel("Mode: rules")
        self._lbl_mode.setObjectName("statChip")
        self._lbl_mode.setAlignment(Qt.AlignCenter)

        layout.addWidget(self._lbl_fps)
        layout.addWidget(self._lbl_mode)
        layout.addSpacing(8)

        footer = QLabel("v2.0  •  Dr. MGR Univ.")
        footer.setObjectName("sidebarFooter")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

        self._go_pending = "translator"  # will be applied after _stack is created
        return sidebar

    def _go(self, key: str):
        page_index = {"translator": 0, "tts": 1, "settings": 2, "about": 3}
        for k, btn in self._nav_btns.items():
            btn.setObjectName("sideButtonActive" if k == key else "sideButton")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self._stack.setCurrentIndex(page_index[key])

    # ── Pages ────────────────────────────────────────────────────────────────

    def _page_translator(self) -> QWidget:
        page = QWidget()
        page.setObjectName("contentFrame")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 28, 36, 20)
        layout.setSpacing(16)

        # Header row
        header_row = QHBoxLayout()
        h_left = QVBoxLayout()
        h_left.setSpacing(2)

        h_title = QLabel("Real-Time Translation")
        h_title.setObjectName("headerTitle")
        h_sub   = QLabel("AI-Powered Live Sign Language Detection")
        h_sub.setObjectName("headerSubtitle")
        h_left.addWidget(h_title)
        h_left.addWidget(h_sub)

        self._emotion_badge = QLabel("😐  NEUTRAL")
        self._emotion_badge.setObjectName("emotionBadge")
        self._emotion_badge.setAlignment(Qt.AlignCenter)
        self._emotion_badge.setFixedHeight(34)

        header_row.addLayout(h_left)
        header_row.addStretch()
        header_row.addWidget(self._emotion_badge)
        layout.addLayout(header_row)

        # Camera
        cam_container = QFrame()
        cam_container.setObjectName("cameraContainer")
        cam_layout = QVBoxLayout(cam_container)
        cam_layout.setContentsMargins(2, 2, 2, 2)

        self._cam_lbl = QLabel()
        self._cam_lbl.setObjectName("cameraView")
        self._cam_lbl.setAlignment(Qt.AlignCenter)
        self._cam_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._cam_lbl.setMinimumHeight(340)
        cam_layout.addWidget(self._cam_lbl)
        layout.addWidget(cam_container, 1)

        # Cards row
        cards_row = QHBoxLayout()
        cards_row.setSpacing(14)

        gest_card, self._gest_lbl = _make_card("DETECTED GESTURE")
        sent_card, self._sent_lbl = _make_card("BUILT SENTENCE")
        self._sent_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        stt_card,  self._stt_lbl  = _make_card("SPEECH INPUT (STT)")

        cards_row.addWidget(gest_card, 1)
        cards_row.addWidget(sent_card, 2)
        cards_row.addWidget(stt_card,  1)
        layout.addLayout(cards_row)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btn_speak = QPushButton("🔊  Speak Now")
        self._btn_speak.setObjectName("primaryButton")
        self._btn_speak.setFixedHeight(40)
        self._btn_speak.setCursor(Qt.PointingHandCursor)
        self._btn_speak.clicked.connect(self._on_speak)

        self._btn_clear = QPushButton("🗑  Clear")
        self._btn_clear.setObjectName("secondaryButton")
        self._btn_clear.setFixedHeight(40)
        self._btn_clear.setCursor(Qt.PointingHandCursor)
        self._btn_clear.clicked.connect(self._on_clear)

        self._status = QLabel("● Initialising…")
        self._status.setObjectName("statusBar")

        btn_row.addWidget(self._btn_speak)
        btn_row.addWidget(self._btn_clear)
        btn_row.addStretch()
        btn_row.addWidget(self._status)
        layout.addLayout(btn_row)

        return page

    def _make_emoji_pixmap(self, emoji: str, size: int = 200) -> QPixmap:
        """Renders an emoji at large size onto a transparent QPixmap."""
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.TextAntialiasing)
        font = QFont("Segoe UI Emoji")
        font.setPixelSize(int(size * 0.7))
        font.setStyleStrategy(QFont.PreferDefault)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))
        from PyQt5.QtCore import QRect
        painter.drawText(QRect(0, 0, size, size),
                         Qt.AlignCenter, emoji)
        painter.end()
        return pixmap

    def _build_tts_panel(self):
        page = QFrame()
        page.setObjectName("contentFrame")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(48, 28, 48, 28)
        layout.setSpacing(16)

        # Header
        header = QLabel("Text  →  Sign + Voice")
        header.setObjectName("headerTitle")
        layout.addWidget(header)

        sub = QLabel("Type a sentence. Each word shows its hand gesture and is spoken aloud.")
        sub.setObjectName("headerSubtitle")
        layout.addWidget(sub)

        # Input row
        input_row = QHBoxLayout()
        input_row.setSpacing(12)

        self.tts_input = QTextEdit()
        self.tts_input.setObjectName("ttsInput")
        self.tts_input.setPlaceholderText("Type here...  e.g.  HELLO YOU GOOD")
        self.tts_input.setFixedHeight(60)
        self.tts_input.setAcceptRichText(False)

        self.tts_go_btn = QPushButton("▶   Translate && Speak")
        self.tts_go_btn.setObjectName("primaryButton")
        self.tts_go_btn.setFixedSize(210, 60)
        self.tts_go_btn.setCursor(Qt.PointingHandCursor)
        self.tts_go_btn.clicked.connect(self._start_tts_playback)

        input_row.addWidget(self.tts_input, 1)
        input_row.addWidget(self.tts_go_btn)
        layout.addLayout(input_row)

        # Progress bar
        self.tts_progress = QProgressBar()
        self.tts_progress.setObjectName("ttsProgress")
        self.tts_progress.setValue(0)
        self.tts_progress.setTextVisible(False)
        self.tts_progress.setFixedHeight(8)
        layout.addWidget(self.tts_progress)

        # Main card
        card = QFrame()
        card.setObjectName("ttsCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 24, 30, 24)
        card_layout.setSpacing(10)
        card_layout.setAlignment(Qt.AlignCenter)

        # Word counter (top)
        self.tts_counter = QLabel("")
        self.tts_counter.setObjectName("ttsCounter")
        self.tts_counter.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.tts_counter)

        # Word label
        self.tts_word_label = QLabel("Waiting for input...")
        self.tts_word_label.setObjectName("ttsWordLabel")
        self.tts_word_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.tts_word_label)

        # Emoji display — this is the main sign visual
        self.tts_emoji_label = QLabel()
        self.tts_emoji_label.setObjectName("ttsEmojiLabel")
        self.tts_emoji_label.setAlignment(Qt.AlignCenter)
        self.tts_emoji_label.setFixedHeight(220)
        # Show default empty hand
        self.tts_emoji_label.setPixmap(
            self._make_emoji_pixmap("🤚", 200))
        card_layout.addWidget(self.tts_emoji_label)

        # Shape description
        self.tts_shape_label = QLabel("Enter a sentence above and press Translate && Speak")
        self.tts_shape_label.setObjectName("ttsShapeLabel")
        self.tts_shape_label.setAlignment(Qt.AlignCenter)
        self.tts_shape_label.setWordWrap(True)
        card_layout.addWidget(self.tts_shape_label)

        layout.addWidget(card, 1)

        # Control buttons
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(12)
        ctrl_row.setAlignment(Qt.AlignCenter)

        self.tts_pause_btn = QPushButton("⏸  Pause")
        self.tts_pause_btn.setObjectName("secondaryButton")
        self.tts_pause_btn.setFixedSize(130, 44)
        self.tts_pause_btn.setCursor(Qt.PointingHandCursor)
        self.tts_pause_btn.setEnabled(False)
        self.tts_pause_btn.clicked.connect(self._pause_tts)

        self.tts_resume_btn = QPushButton("▶  Resume")
        self.tts_resume_btn.setObjectName("secondaryButton")
        self.tts_resume_btn.setFixedSize(130, 44)
        self.tts_resume_btn.setCursor(Qt.PointingHandCursor)
        self.tts_resume_btn.setEnabled(False)
        self.tts_resume_btn.clicked.connect(self._resume_tts)

        self.tts_stop_btn = QPushButton("⏹  Stop")
        self.tts_stop_btn.setObjectName("secondaryButton")
        self.tts_stop_btn.setFixedSize(130, 44)
        self.tts_stop_btn.setCursor(Qt.PointingHandCursor)
        self.tts_stop_btn.setEnabled(False)
        self.tts_stop_btn.clicked.connect(self._stop_tts)

        ctrl_row.addWidget(self.tts_pause_btn)
        ctrl_row.addWidget(self.tts_resume_btn)
        ctrl_row.addWidget(self.tts_stop_btn)
        layout.addLayout(ctrl_row)

        return page

    def _start_tts_playback(self):
        """Called when user clicks Translate & Speak."""
        text = self.tts_input.toPlainText().strip()
        if not text:
            self.tts_word_label.setText("Please type something first!")
            return

        # Stop any existing playback
        if self._tts_worker and self._tts_worker.isRunning():
            self._tts_worker.stop_playback()
            self._tts_worker.wait()

        words = text.split()
        self.tts_progress.setMaximum(len(words))
        self.tts_progress.setValue(0)

        self._tts_worker = TextToSignWorker(words, interval=2.5)
        self._tts_worker.word_signal.connect(self._on_tts_word)
        self._tts_worker.done_signal.connect(self._on_tts_done)
        self._tts_worker.start()

        self.tts_go_btn.setEnabled(False)
        self.tts_pause_btn.setEnabled(True)
        self.tts_resume_btn.setEnabled(False)
        self.tts_stop_btn.setEnabled(True)

    def _on_tts_word(self, word, shape, index, total):
        GESTURE_EMOJI = {
            "NO":        ("✊", "Closed fist — all fingers curled"),
            "HELLO":     ("🖐️", "Open palm — all 5 fingers up"),
            "YES":       ("👍", "Thumbs up only"),
            "ME":        ("☝️", "Index finger only — point at yourself"),
            "YOU":       ("✌️", "Peace sign — Index + Middle up"),
            "STOP":      ("🤚", "4 fingers up — thumb tucked in"),
            "GOOD":      ("👌", "Thumb + Index — L shape"),
            "CALL ME":   ("🤙", "Thumb + Pinky — phone sign"),
            "AGAIN":     ("🤘", "Index + Pinky — rock horns"),
            "HOW":       ("🖖", "Thumb + Index + Middle up"),
            "HELP":      ("🤟", "Index + Middle + Pinky up"),
            "WE":        ("🤞", "Index + Middle + Ring up"),
            "FINISH":    ("🖐️", "All fingers up except Pinky"),
            "GIVE":      ("🫴", "Thumb + Index + Pinky up"),
            "WAIT":      ("✋", "Thumb + Index + Ring up"),
            "PLEASE":    ("🙏", "Both palms together"),
            "COME":      ("👋", "Thumb + Middle only"),
            "SORRY":     ("✊", "Thumb + Ring only"),
            "WHAT":      ("🤷", "Thumb + Ring + Pinky up"),
            "WHERE":     ("👐", "Middle + Ring + Pinky up"),
            "KNOW":      ("🫵", "Middle + Ring only"),
            "NEED":      ("🤲", "Ring + Pinky only"),
            "WANT":      ("💅", "Ring finger only"),
            "HE/SHE":    ("🤙", "Pinky only"),
            "SEE":       ("👀", "Middle + Pinky up"),
            "THANK YOU": ("🙌", "Index + Ring up"),
            "GO":        ("👉", "Index + Ring + Pinky up"),
            "NOW":       ("👇", "Thumb + Middle + Pinky up"),
            "BAD":       ("👎", "Thumb + Middle + Ring up"),
            "TAKE":      ("🫳", "Thumb + Middle + Ring + Pinky up"),
            "LATER":     ("🤚", "Thumb + Index + Middle + Pinky up"),
            "OK":        ("🖕", "Middle only up"),
        }
        emoji, description = GESTURE_EMOJI.get(
            word.upper(), ("✋", shape))

        self.tts_counter.setText(f"Word {index} of {total}")
        self.tts_word_label.setText(word.upper())
        self.tts_shape_label.setText(description)
        self.tts_progress.setValue(index)
        self.tts_emoji_label.setPixmap(
            self._make_emoji_pixmap(emoji, 200))

    def _on_tts_done(self):
        """Called when all words are done."""
        self.tts_word_label.setText("✅  Complete!")
        self.tts_shape_label.setText("All words translated. Type a new sentence above.")
        self.tts_emoji_label.setPixmap(self._make_emoji_pixmap("✅", 200))
        self.tts_counter.setText("")
        self.tts_progress.setValue(self.tts_progress.maximum())
        self.tts_go_btn.setEnabled(True)
        self.tts_pause_btn.setEnabled(False)
        self.tts_resume_btn.setEnabled(False)
        self.tts_stop_btn.setEnabled(False)

    def _pause_tts(self):
        if self._tts_worker:
            self._tts_worker.pause()
            self.tts_pause_btn.setEnabled(False)
            self.tts_resume_btn.setEnabled(True)
            self.tts_shape_label.setText("⏸  Paused — click Resume to continue")

    def _resume_tts(self):
        if self._tts_worker:
            self._tts_worker.resume()
            self.tts_pause_btn.setEnabled(True)
            self.tts_resume_btn.setEnabled(False)

    def _stop_tts(self):
        if self._tts_worker:
            self._tts_worker.stop_playback()
            self._tts_worker.wait()
        self.tts_word_label.setText("⏹  Stopped")
        self.tts_shape_label.setText("Type a new sentence above and click Translate & Speak")
        self.tts_emoji_label.setPixmap(self._make_emoji_pixmap("⏹️", 200))
        self.tts_counter.setText("")
        self.tts_progress.setValue(0)
        self.tts_go_btn.setEnabled(True)
        self.tts_pause_btn.setEnabled(False)
        self.tts_resume_btn.setEnabled(False)
        self.tts_stop_btn.setEnabled(False)
    def _page_settings(self) -> QWidget:
        page = QWidget()
        page.setObjectName("contentFrame")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(48, 36, 48, 36)
        layout.setSpacing(20)

        layout.addWidget(QLabel("Settings", objectName="headerTitle"))
        layout.addWidget(_separator())

        # Hold time slider
        self._lbl_hold = QLabel(f"Gesture Hold Time:  {self._rec.HOLD_TIME:.1f} s")
        self._lbl_hold.setObjectName("settingLabel")
        layout.addWidget(self._lbl_hold)

        sld_hold = QSlider(Qt.Horizontal)
        sld_hold.setRange(3, 20)
        sld_hold.setValue(int(self._rec.HOLD_TIME * 10))
        sld_hold.valueChanged.connect(self._set_hold)
        layout.addWidget(sld_hold)

        # Speak delay slider
        self._lbl_speak = QLabel(f"Auto-Speak Delay:  {self._rec.SPEAK_DELAY:.1f} s")
        self._lbl_speak.setObjectName("settingLabel")
        layout.addWidget(self._lbl_speak)

        sld_speak = QSlider(Qt.Horizontal)
        sld_speak.setRange(10, 80)
        sld_speak.setValue(int(self._rec.SPEAK_DELAY * 10))
        sld_speak.valueChanged.connect(self._set_speak_delay)
        layout.addWidget(sld_speak)

        layout.addWidget(_separator())

        # Checkboxes
        self._chk_tts = QCheckBox("Enable Text-to-Speech")
        self._chk_tts.setObjectName("settingCheck")
        self._chk_tts.setChecked(True)
        self._chk_tts.stateChanged.connect(self._toggle_tts)
        layout.addWidget(self._chk_tts)

        self._chk_stt = QCheckBox("Enable Speech-to-Text (microphone)")
        self._chk_stt.setObjectName("settingCheck")
        self._chk_stt.setChecked(False)
        self._chk_stt.stateChanged.connect(self._toggle_stt)
        layout.addWidget(self._chk_stt)

        self._chk_emotion = QCheckBox("Enable Emotion Detection")
        self._chk_emotion.setObjectName("settingCheck")
        self._chk_emotion.setChecked(True)
        self._chk_emotion.stateChanged.connect(lambda s: setattr(self, "_emotion_on", bool(s)))
        layout.addWidget(self._chk_emotion)

        layout.addStretch()
        return page

    def _page_about(self) -> QWidget:
        page = QWidget()
        page.setObjectName("contentFrame")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(48, 36, 48, 36)
        layout.setSpacing(16)

        layout.addWidget(QLabel("About SignTalk", objectName="headerTitle"))
        layout.addWidget(_separator())

        about_html = """
        <p><b>SignTalk</b> is a real-time AI-powered Sign Language Translator that
        bridges communication between the hearing-impaired and the hearing world.</p>
        <br>
        <p><b>Authors:</b><br>
        &nbsp;&nbsp;• Dinesh Kumar S<br>
        &nbsp;&nbsp;• Diwakar</p>
        <br>
        <p><b>Guide:</b> Dr. V. Rameshbabu</p>
        <br>
        <p><b>Institution:</b><br>
        Dr. M.G.R. Educational and Research Institute</p>
        <br>
        <p><b>Tech Stack:</b><br>
        &nbsp;&nbsp;• Python · OpenCV · MediaPipe<br>
        &nbsp;&nbsp;• TensorFlow / Keras (CNN-LSTM)<br>
        &nbsp;&nbsp;• PyQt5 / PySide6 · pyttsx3<br>
        &nbsp;&nbsp;• SpeechRecognition / Vosk (STT)</p>
        """
        about_lbl = QLabel(about_html)
        about_lbl.setObjectName("aboutText")
        about_lbl.setWordWrap(True)
        layout.addWidget(about_lbl)

        layout.addStretch()
        return page

    # ── Styles ────────────────────────────────────────────────────────────────

    def _load_styles(self):
        qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        else:
            print(f"[Warning] style.qss not found at {qss_path}")

    # ── Frame update ──────────────────────────────────────────────────────────

    def _update_frame(self):
        ret, frame = self._cap.read()
        if not ret:
            self._status.setText("⚠  Camera disconnected")
            return

        frame = cv2.flip(frame, 1)
        frame, gesture, sentence, emotion = self._rec.process_frame(frame)

        # Convert to QPixmap – keep QImage alive to prevent GC segfault
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._last_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        self._cam_lbl.setPixmap(
            QPixmap.fromImage(self._last_img).scaled(
                self._cam_lbl.width(),
                self._cam_lbl.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        # Labels
        self._gest_lbl.setText(gesture or "—")
        self._sent_lbl.setText(sentence or "—")

        # Emotion badge
        if self._emotion_on:
            icon  = _EMOTION_ICON.get(emotion, "😐")
            color = _EMOTION_COLOR.get(emotion, "#94a3b8")
            self._emotion_badge.setText(f"{icon}  {emotion}")
            self._emotion_badge.setStyleSheet(
                f"color: {color}; font-weight: bold; font-size: 15px;"
            )

        # FPS
        now = time.time()
        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t < 1.0]
        self._lbl_fps.setText(f"FPS: {len(self._fps_times)}")

        # Mode chip
        self._lbl_mode.setText(
            "Mode: CNN-LSTM" if self._rec.model_loaded else "Mode: rules"
        )

        # Status bar
        self._status.setText(
            f"● Gesture: {gesture}" if gesture else "● Monitoring…"
        )

    # ── Button handlers ───────────────────────────────────────────────────────

    def _on_speak(self):
        self._rec.speak_now()
        self._status.setText("🔊  Speaking…")

    def _on_clear(self):
        self._rec.clear_sentence()
        self._sent_lbl.setText("—")
        self._status.setText("🗑  Cleared")

    # ── Settings handlers ─────────────────────────────────────────────────────

    def _set_hold(self, v: int):
        self._rec.HOLD_TIME = v / 10
        self._lbl_hold.setText(f"Gesture Hold Time:  {self._rec.HOLD_TIME:.1f} s")

    def _set_speak_delay(self, v: int):
        self._rec.SPEAK_DELAY = v / 10
        self._lbl_speak.setText(f"Auto-Speak Delay:  {self._rec.SPEAK_DELAY:.1f} s")

    def _toggle_tts(self, state: int):
        self._rec.SPEAK_DELAY = 3.5 if state else 99999

    def _toggle_stt(self, state: int):
        if state == 2:
            self._stt.start()
            self._stt_lbl.setText("Listening…")
        else:
            self._stt.stop()
            self._stt_lbl.setText("— (disabled)")

    def _on_speech(self, text: str):
        self._stt_lbl.setText(f'"{text}"')
        self._status.setText(f"🎙  Heard: {text}")

    # ── Close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._tts_worker and self._tts_worker.isRunning():
            self._tts_worker.stop_playback()
            self._tts_worker.wait()

        self._timer.stop()   # ← must be first to prevent segfault
        self._stt.stop()
        self._cap.release()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("SignTalk")
    win = SignTranslatorUI()
    win.show()
    sys.exit(app.exec_() if _QT == "PyQt5" else app.exec())