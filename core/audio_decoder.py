import struct
import logging

import numpy as np
import opuslib

logger = logging.getLogger(__name__)

# Must match the OBS plugin encoder settings (remote_transcriber.h)
SAMPLE_RATE = 16000   # Hz
CHANNELS = 1          # mono
FRAME_SIZE = 320      # samples per frame = 20ms at 16kHz

# Maximum consecutive Opus decode failures before the decoder is reset
_MAX_CONSECUTIVE_ERRORS = 3


class AudioMessage:
    """Parsed binary message from OBS plugin."""

    def __init__(self, sentence_id: int, is_final: bool, audio_pcm: np.ndarray):
        self.sentence_id = sentence_id
        self.is_final = is_final
        self.audio_pcm = audio_pcm


class OpusAudioDecoder:
    """
    Decodes the binary Opus protocol sent by the OBS plugin.

    Binary message layout (see remote_transcriber.h):
      Bytes 0-3  : sentence_id (uint32, little-endian)
      Byte  4    : is_final    (0 = partial, 1 = final)
      Bytes 5+   : Opus frames, each prefixed with its length:
                     [2 bytes length LE][N bytes Opus data]

    Robustness
    ----------
    If more than _MAX_CONSECUTIVE_ERRORS Opus frames fail in a single message,
    the internal decoder state is reset to prevent carrying corruption into
    the next message. The reset is transparent to the caller.
    """

    def __init__(self):
        self._decoder = opuslib.Decoder(SAMPLE_RATE, CHANNELS)

    def reset(self) -> None:
        """
        Recreate the internal Opus decoder to clear any corrupted state.
        Called automatically after too many consecutive decode errors.
        """
        logger.warning("[OpusDecoder] Resetting decoder due to repeated errors.")
        self._decoder = opuslib.Decoder(SAMPLE_RATE, CHANNELS)

    def decode_message(self, data: bytes) -> AudioMessage:
        """Decode a single binary WebSocket message into an AudioMessage."""
        if len(data) < 5:
            raise ValueError(f"Message too short: {len(data)} bytes (minimum 5)")

        # ── Parse header ──────────────────────────────────────────────────
        sentence_id = struct.unpack_from('<I', data, 0)[0]
        is_final = bool(data[4])

        # ── Decode Opus frames ────────────────────────────────────────────
        offset = 5
        pcm_frames: list[np.ndarray] = []
        consecutive_errors = 0

        while offset + 2 <= len(data):
            frame_len = struct.unpack_from('<H', data, offset)[0]
            offset += 2

            if frame_len == 0 or offset + frame_len > len(data):
                break

            opus_data = data[offset:offset + frame_len]
            offset += frame_len

            try:
                # opuslib.Decoder.decode() returns raw bytes of int16 PCM
                pcm_bytes = self._decoder.decode(opus_data, FRAME_SIZE)
                pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                pcm_frames.append(pcm_int16)
                consecutive_errors = 0  # Reset counter on success
            except opuslib.OpusError as e:
                consecutive_errors += 1
                logger.warning(
                    f"[OpusDecoder] Frame decode error ({consecutive_errors}/"
                    f"{_MAX_CONSECUTIVE_ERRORS}): {e}"
                )
                if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    self.reset()
                    consecutive_errors = 0
                continue

        if not pcm_frames:
            raise ValueError(
                f"No valid Opus frames decoded from message "
                f"(sentence_id={sentence_id}, data_len={len(data)})"
            )

        # Concatenate all frames and normalize to float32 [-1.0, 1.0]
        audio_int16 = np.concatenate(pcm_frames)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        return AudioMessage(sentence_id, is_final, audio_float32)
