import struct
import pytest
import numpy as np
import opuslib

from core.audio_decoder import OpusAudioDecoder, AudioMessage, FRAME_SIZE, SAMPLE_RATE, CHANNELS


def create_mock_opus_message(sentence_id: int, is_final: bool, num_frames: int = 5) -> bytes:
    """Helper to create a binary payload with valid Opus frames."""
    encoder = opuslib.Encoder(SAMPLE_RATE, CHANNELS, opuslib.APPLICATION_VOIP)
    
    header = struct.pack('<IB', sentence_id, 1 if is_final else 0)
    body = bytearray()
    
    # Simple sine wave frame (320 samples = 20ms at 16kHz)
    t = np.linspace(0, 0.02, FRAME_SIZE, endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)
    pcm_bytes = pcm.tobytes()
    
    for _ in range(num_frames):
        opus_frame = encoder.encode(pcm_bytes, FRAME_SIZE)
        frame_len = len(opus_frame)
        body.extend(struct.pack('<H', frame_len))
        body.extend(opus_frame)
        
    return header + bytes(body)


def test_decode_valid_message():
    decoder = OpusAudioDecoder()
    raw_payload = create_mock_opus_message(sentence_id=42, is_final=True, num_frames=3)
    
    msg = decoder.decode_message(raw_payload)
    
    assert isinstance(msg, AudioMessage)
    assert msg.sentence_id == 42
    assert msg.is_final is True
    assert msg.audio_pcm.dtype == np.float32
    assert len(msg.audio_pcm) == 3 * FRAME_SIZE  # 3 frames * 320 samples = 960 samples
    assert np.max(np.abs(msg.audio_pcm)) <= 1.0


def test_decode_message_too_short():
    decoder = OpusAudioDecoder()
    short_data = b'\x01\x00\x00\x00'  # 4 bytes (needs >= 5)
    with pytest.raises(ValueError, match="Message too short"):
        decoder.decode_message(short_data)


def test_decode_message_corrupted_frame():
    decoder = OpusAudioDecoder()
    header = struct.pack('<IB', 1, 0)
    corrupted_frame = struct.pack('<H', 10) + b'X' * 10
    
    with pytest.raises(ValueError, match="No valid Opus frames decoded"):
        decoder.decode_message(header + corrupted_frame)
