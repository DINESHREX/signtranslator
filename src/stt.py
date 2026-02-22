import time
import threading

# ── Optional dependency probes ────────────────────────────────────────────────
try:
    import speech_recognition as sr
    _SR_OK = True
except ImportError:
    _SR_OK = False

try:
    import vosk
    import json as _json
    import pyaudio
    _VOSK_OK = True
except ImportError:
    _VOSK_OK = False


class SpeechListener:
    """
    Non-blocking speech-to-text listener.

    Backends (tried in order):
      1. Google Web Speech API  (requires: SpeechRecognition, pyaudio)
      2. Vosk offline model     (requires: vosk, pyaudio)
      3. None – prints install hint
    """

    def __init__(
        self,
        callback=None,
        language: str = "en-US",
        vosk_model_path: str = "models/vosk-model-small-en"
    ):
        self._cb        = callback or (lambda t: print(f"[STT] {t}"))
        self._lang      = language
        self._vosk_path = vosk_model_path
        self._running   = False
        self._thread    = None

        if _SR_OK:
            self._mode       = "google"
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold        = 300
            self._recognizer.dynamic_energy_threshold = True
            self._mic = sr.Microphone()
            print("[STT] Backend: Google Web Speech  (SpeechRecognition)")

        elif _VOSK_OK:
            self._mode = "vosk"
            print(f"[STT] Backend: Vosk offline  (model: {vosk_model_path})")

        else:
            self._mode = "none"
            print(
                "[STT] Warning: no STT backend available.\n"
                "      Install SpeechRecognition and pyaudio for Google STT:\n"
                "        pip install SpeechRecognition pyaudio\n"
                "      OR install Vosk and pyaudio for offline STT:\n"
                "        pip install vosk pyaudio"
            )

    # ── Internal run loops ────────────────────────────────────────────────────

    def _run_google(self):
        """Google Web Speech API loop (blocking, runs in daemon thread)."""
        with self._mic as source:
            self._recognizer.adjust_for_ambient_noise(source, duration=1.0)

        while self._running:
            try:
                with self._mic as source:
                    audio = self._recognizer.listen(
                        source,
                        timeout=4,
                        phrase_time_limit=8
                    )
                text = self._recognizer.recognize_google(audio, language=self._lang)
                self._cb(text.upper())

            except sr.WaitTimeoutError:
                pass  # silence – keep listening
            except sr.UnknownValueError:
                pass  # unintelligible audio
            except sr.RequestError:
                time.sleep(3)  # network / API error
            except Exception:
                time.sleep(1)

    def _run_vosk(self):
        """Vosk offline STT loop (blocking, runs in daemon thread)."""
        model      = vosk.Model(self._vosk_path)
        recognizer = vosk.KaldiRecognizer(model, 16000)

        pa     = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8192
        )

        try:
            while self._running:
                data = stream.read(4096, exception_on_overflow=False)
                if recognizer.AcceptWaveform(data):
                    result = _json.loads(recognizer.Result())
                    text   = result.get("text", "").strip()
                    if text:
                        self._cb(text.upper())
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start the background listening thread. No-op if already running."""
        if self._running or self._mode == "none":
            return

        self._running = True
        target = self._run_google if self._mode == "google" else self._run_vosk
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the listener to stop and wait up to 3 s for it to finish."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    @property
    def active(self) -> bool:
        """True if the listener is running and the thread is alive."""
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def mode(self) -> str:
        """Active backend: 'google', 'vosk', or 'none'."""
        return self._mode
