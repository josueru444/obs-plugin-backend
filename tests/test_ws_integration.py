import os
import struct
import pytest
import numpy as np
import opuslib
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from main import app
from core.audio_decoder import FRAME_SIZE, SAMPLE_RATE, CHANNELS


def get_auth_token():
    return os.getenv("OBS_API_KEY", "obs_secret_key")


def create_test_opus_bytes(sentence_id: int = 1, is_final: bool = True) -> bytes:
    encoder = opuslib.Encoder(SAMPLE_RATE, CHANNELS, opuslib.APPLICATION_VOIP)
    header = struct.pack('<IB', sentence_id, 1 if is_final else 0)
    
    t = np.linspace(0, 0.02, FRAME_SIZE, endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)
    opus_frame = encoder.encode(pcm.tobytes(), FRAME_SIZE)
    
    body = struct.pack('<H', len(opus_frame)) + opus_frame
    return header + body


def test_ws_rejects_invalid_lang_params():
    client = TestClient(app)
    token = get_auth_token()
    with pytest.raises(Exception):
        with client.websocket_connect(f"/ws/audio/?token={token}&lang_in=invalid_lang") as websocket:
            pass


@patch("api.endpoints_ws.get_whisper_service")
def test_ws_stream_audio_and_receive_response(mock_get_whisper):
    # Mock Whisper service to simulate instant streaming callback
    mock_whisper = AsyncMock()
    
    async def fake_transcribe(audio_data, on_segment, language, task, is_final_message):
        await on_segment("Transcribed test speech", is_final_message)

    mock_whisper.transcribe_streaming = fake_transcribe
    mock_get_whisper.return_value = mock_whisper

    client = TestClient(app)
    token = get_auth_token()
    opus_data = create_test_opus_bytes(sentence_id=101, is_final=True)

    with client.websocket_connect(f"/ws/audio/?token={token}&lang_in=es&lang_out=original") as websocket:
        websocket.send_bytes(opus_data)
        data = websocket.receive_json()

        assert data["sentence_id"] == 101
        assert data["is_final"] is True
        assert data["text"] == "Transcribed test speech"
