"""
Real-time Audio Stream Simulator for OBS Translator Backend.
Replicates the binary WebSocket communication of the OBS C++ plugin (`obs-plugin-traduccion`).

Protocol specification:
  Header:
    - Bytes 0..3 : sentence_id (uint32 LE)
    - Byte  4    : is_final    (uint8: 0 = partial, 1 = final)
  Opus Payload:
    - [2 bytes length LE][N bytes Opus data] ... per 20ms frame (320 samples @ 16kHz mono)
"""

import asyncio
import argparse
import logging
import struct
import time
import sys
import numpy as np
import opuslib
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OBS-Audio-Simulator")

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SIZE = 320  # 20ms at 16kHz


class RealtimeAudioSimulator:
    def __init__(self, uri: str):
        self.uri = uri
        self.encoder = opuslib.Encoder(SAMPLE_RATE, CHANNELS, opuslib.APPLICATION_VOIP)
        self.encoder.bitrate = 24000

        self.sent_count = 0
        self.received_count = 0
        self.latencies = []

    def generate_speech_audio(self, duration_sec: float, freq: float = 440.0) -> np.ndarray:
        """Generates synthetic multi-frequency speech-like audio signal."""
        t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
        # Fundamental frequency + harmonics + amplitude modulation to mimic speech envelope
        signal = (
            0.5 * np.sin(2 * np.pi * freq * t) +
            0.25 * np.sin(2 * np.pi * freq * 2 * t) +
            0.125 * np.sin(2 * np.pi * freq * 3 * t)
        )
        envelope = np.sin(np.pi * t / duration_sec)  # Smooth onset/offset
        speech = (signal * envelope * 16384).astype(np.int16)
        return speech

    def encode_segment(self, sentence_id: int, pcm_int16: np.ndarray, is_final: bool) -> bytes:
        """Encodes PCM audio into the C++ OBS plugin binary format."""
        header = struct.pack('<IB', sentence_id, 1 if is_final else 0)
        body = bytearray()

        offset = 0
        total_samples = len(pcm_int16)

        while offset + FRAME_SIZE <= total_samples:
            frame_pcm = pcm_int16[offset:offset + FRAME_SIZE].tobytes()
            opus_frame = self.encoder.encode(frame_pcm, FRAME_SIZE)
            body.extend(struct.pack('<H', len(opus_frame)))
            body.extend(opus_frame)
            offset += FRAME_SIZE

        # Pad remaining samples if final segment
        if is_final and offset < total_samples:
            padded = np.zeros(FRAME_SIZE, dtype=np.int16)
            rem = total_samples - offset
            padded[:rem] = pcm_int16[offset:]
            opus_frame = self.encoder.encode(padded.tobytes(), FRAME_SIZE)
            body.extend(struct.pack('<H', len(opus_frame)))
            body.extend(opus_frame)

        return header + bytes(body)

    async def receive_loop(self, ws: websockets.WebSocketClientProtocol, stop_event: asyncio.Event):
        """Receives JSON responses from the backend WebSocket server."""
        try:
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    self.received_count += 1
                    recv_time = time.time()
                    logger.info(f"🟢 [BACKEND RESPONSE #{self.received_count}] -> {msg}")
                except asyncio.TimeoutError:
                    continue
        except websockets.ConnectionClosed:
            logger.info("WebSocket connection closed by server.")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")

    async def run_simulation(self, sentences_to_send: int = 3):
        """Simulates real-time audio streaming of multiple spoken sentences."""
        logger.info(f"Connecting to WebSocket target: {self.uri} ...")

        async with websockets.connect(self.uri) as ws:
            logger.info("Connected successfully! Starting real-time audio streaming simulation...\n")
            stop_event = asyncio.Event()
            recv_task = asyncio.create_task(self.receive_loop(ws, stop_event))

            for sid in range(1, sentences_to_send + 1):
                logger.info(f"--- Streaming Sentence {sid}/{sentences_to_send} ---")
                
                # Simulate speech sentence of 3 seconds: 5 partial updates + 1 final update
                partial_durations = [0.5, 1.0, 1.5, 2.0, 2.5]
                final_duration = 3.0

                start_time = time.time()

                # Send partial segments every 500ms
                for dur in partial_durations:
                    pcm = self.generate_speech_audio(dur, freq=220 + sid * 50)
                    payload = self.encode_segment(sentence_id=sid, pcm_int16=pcm, is_final=False)
                    await ws.send(payload)
                    self.sent_count += 1
                    logger.info(f"  [SENT PARTIAL] sid={sid}, duration={dur:.1f}s, bytes={len(payload)}")
                    await asyncio.sleep(0.5)

                # Send final segment
                pcm_final = self.generate_speech_audio(final_duration, freq=220 + sid * 50)
                payload_final = self.encode_segment(sentence_id=sid, pcm_int16=pcm_final, is_final=True)
                send_t = time.time()
                await ws.send(payload_final)
                self.sent_count += 1
                logger.info(f"  [SENT FINAL]   sid={sid}, duration={final_duration:.1f}s, bytes={len(payload_final)}")
                
                elapsed = time.time() - start_time
                self.latencies.append(time.time() - send_t)

                # Pause between sentences (silence simulation)
                await asyncio.sleep(1.5)

            # Wait for pending responses
            await asyncio.sleep(2.0)
            stop_event.set()
            await recv_task

        logger.info("================ SIMULATION SUMMARY ================")
        logger.info(f"Total Sent Messages     : {self.sent_count}")
        logger.info(f"Total Received Responses: {self.received_count}")
        logger.info("====================================================")


async def main():
    parser = argparse.ArgumentParser(description="Simulate real-time OBS plugin audio streaming")
    parser.add_argument("--url", default="ws://localhost:8000/ws/audio/?lang_in=es&lang_out=original", help="WebSocket URL")
    parser.add_argument("--sentences", type=int, default=3, help="Number of sentences to stream")
    args = parser.parse_args()

    sim = RealtimeAudioSimulator(args.url)
    await sim.run_simulation(sentences_to_send=args.sentences)


if __name__ == "__main__":
    asyncio.run(main())
