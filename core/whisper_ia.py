import asyncio
import logging
import os
import threading
from typing import Callable, Awaitable

import numpy as np
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format='%(levelname)s:     %(message)s')
logger = logging.getLogger(__name__)


class WhisperService:
    """
    Whisper inference service backed by whisper.cpp via pywhispercpp.

    GPU backend (Vulkan / CUDA) is selected at INSTALL TIME:
      - Vulkan: GGML_VULKAN=1 pip install git+https://github.com/absadiki/pywhispercpp
      - CUDA:   GGML_CUDA=1   pip install git+https://github.com/absadiki/pywhispercpp
      - CPU:    pip install pywhispercpp  (default PyPI wheel)

    At runtime:
      - WHISPER_USE_GPU=true/false enables or disables GPU context.
      - WHISPER_THREADS controls CPU worker threads.

    Concurrency
    -----------
    whisper.cpp's model object is NOT thread-safe. A single asyncio.Semaphore(1)
    serializes all inference calls so that only one transcription runs at a time.
    This is safe for the current single-client use case and prevents any race
    condition on internal model state.
    """

    def __init__(self, model_name: str = "base", n_threads: int = 4, use_gpu: bool = True):
        logger.info(
            f"Loading whisper.cpp model '{model_name}' "
            f"(threads={n_threads}, use_gpu={use_gpu})..."
        )
        self.model = WhisperModel(
            model_size_or_path=model_name,
            device="cuda" if use_gpu else "cpu",
            compute_type="float16" if use_gpu else "int8",
            cpu_threads=n_threads,
        )
        # Serializes access to the whisper.cpp model — it is NOT thread-safe.
        self._semaphore = asyncio.Semaphore(1)
        logger.info("whisper.cpp model loaded successfully.")

    @property
    def is_busy(self) -> bool:
        """
        Returns True if a transcription is currently running.
        Used by the WebSocket endpoint to discard stale partial segments
        instead of queuing them behind an already-running inference.
        """
        return self._semaphore.locked()

    async def transcribe_streaming(
        self,
        audio_data: np.ndarray,
        on_segment: Callable[[str, bool], Awaitable[None]],
        language: str = "es",
        task: str = "transcribe",
        is_final_message: bool = True,
    ) -> None:
        """
        Transcribes audio and streams segments as they are produced by whisper.cpp.

        For each Whisper segment decoded:
          - All segments except the last are sent as partial (is_final=False).
          - The last segment is sent with is_final matching the original OBS message flag.

        :param audio_data:        Raw PCM float32 at 16kHz mono.
        :param on_segment:        Async callback invoked for each decoded segment.
                                  Receives (text: str, is_final: bool).
        :param language:          Source language code or 'auto'.
        :param task:              'transcribe' or 'translate'.
        :param is_final_message:  Whether the OBS message was marked as final.
        """
        # whisper.cpp requires an empty string for auto language detection,
        # not the literal string "auto".
        lang = language if language and language not in ("auto", "original", "") else ""
        translate = task == "translate"

        loop = asyncio.get_event_loop()
        collected: list[str] = []

        def _run_transcription() -> None:
            logger.info(f"[Whisper] transcribe → language='{lang}', translate={translate}")
            segments, _ = self.model.transcribe(
                audio_data,
                language=lang if lang else None,
                task="translate" if translate else "transcribe",
                beam_size=1,
                vad_filter=True,
            )
            for segment in segments:
                text = segment.text.strip()
                if text:
                    collected.append(text)

        # Acquire the semaphore BEFORE entering the thread pool.
        # This guarantees that only one transcription runs at a time,
        # preventing race conditions on the shared model state.
        async with self._semaphore:
            await asyncio.to_thread(_run_transcription)

        if not collected:
            return

        # Stream collected segments back to the WebSocket caller.
        # We concatenate all segments into a single string representing the
        # full transcription of the current audio buffer.
        full_text = " ".join(collected)
        await on_segment(full_text, is_final_message)


# ─── Singleton ────────────────────────────────────────────────────────────────

_whisper_service: WhisperService | None = None
_whisper_lock = threading.Lock()


def get_whisper_service() -> WhisperService:
    """
    Return the global WhisperService instance (singleton).

    Uses a threading.Lock to prevent a double-initialization race condition
    on startup when the server first handles requests concurrently.
    """
    global _whisper_service

    if _whisper_service is not None:
        return _whisper_service

    with _whisper_lock:
        # Double-checked locking: re-check after acquiring the lock
        # in case another thread already initialized it.
        if _whisper_service is None:
            model_name = os.getenv("WHISPER_MODEL", "base")
            n_threads = int(os.getenv("WHISPER_THREADS", "4"))
            use_gpu = os.getenv("WHISPER_USE_GPU", "true").lower() in ("true", "1", "yes")

            _whisper_service = WhisperService(
                model_name=model_name,
                n_threads=n_threads,
                use_gpu=use_gpu,
            )

    return _whisper_service
