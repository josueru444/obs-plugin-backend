"""
Performance & Stress Benchmark for OBS Translator Backend.
Tests concurrent WebSocket connections, latency, throughput, and memory consumption.
"""

import asyncio
import time
import logging
import psutil
import os
import struct
import numpy as np
import opuslib
import websockets

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SIZE = 320

logger = logging.getLogger("Backend-Stress-Tester")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def create_opus_payload(sentence_id: int, is_final: bool, duration_sec: float = 1.0) -> bytes:
    encoder = opuslib.Encoder(SAMPLE_RATE, CHANNELS, opuslib.APPLICATION_VOIP)
    header = struct.pack('<IB', sentence_id, 1 if is_final else 0)
    body = bytearray()

    num_samples = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)
    pcm_bytes = pcm.tobytes()

    offset = 0
    while offset + FRAME_SIZE <= len(pcm):
        frame = pcm[offset:offset + FRAME_SIZE].tobytes()
        opus_frame = encoder.encode(frame, FRAME_SIZE)
        body.extend(struct.pack('<H', len(opus_frame)))
        body.extend(opus_frame)
        offset += FRAME_SIZE

    return header + bytes(body)


async def single_client_worker(client_id: int, uri: str, num_sentences: int, results: list):
    sent_msgs = 0
    recv_msgs = 0
    latencies = []

    try:
        async with websockets.connect(uri) as ws:
            for sid in range(1, num_sentences + 1):
                payload = create_opus_payload(sentence_id=sid, is_final=True, duration_sec=1.0)
                t0 = time.time()
                await ws.send(payload)
                sent_msgs += 1

                try:
                    res = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    rtt = time.time() - t0
                    latencies.append(rtt)
                    recv_msgs += 1
                except asyncio.TimeoutError:
                    logger.warning(f"Client #{client_id} timed out waiting for sentence {sid}")

                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Client #{client_id} error: {e}")

    results.append({
        "client_id": client_id,
        "sent": sent_msgs,
        "recv": recv_msgs,
        "avg_latency": np.mean(latencies) if latencies else 0.0,
        "max_latency": np.max(latencies) if latencies else 0.0,
    })


async def run_stress_test(uri: str, num_clients: int, num_sentences: int):
    logger.info(f"Starting Stress Test: {num_clients} concurrent clients streaming to {uri}")

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    t_start = time.time()
    results = []
    tasks = [
        single_client_worker(client_id=i, uri=uri, num_sentences=num_sentences, results=results)
        for i in range(1, num_clients + 1)
    ]
    await asyncio.gather(*tasks)
    total_time = time.time() - t_start

    mem_after = process.memory_info().rss / (1024 * 1024)

    total_sent = sum(r["sent"] for r in results)
    total_recv = sum(r["recv"] for r in results)
    all_avg_latencies = [r["avg_latency"] for r in results if r["avg_latency"] > 0]

    logger.info("================ STRESS TEST RESULTS ================")
    logger.info(f"Concurrent Clients     : {num_clients}")
    logger.info(f"Total Execution Time   : {total_time:.2f} s")
    logger.info(f"Total Messages Sent    : {total_sent}")
    logger.info(f"Total Responses Recv   : {total_recv}")
    logger.info(f"Overall Avg Latency    : {np.mean(all_avg_latencies):.3f} s" if all_avg_latencies else "N/A")
    logger.info(f"Memory RSS Start       : {mem_before:.2f} MB")
    logger.info(f"Memory RSS End         : {mem_after:.2f} MB")
    logger.info(f"Memory Delta           : {mem_after - mem_before:+.2f} MB")
    logger.info("=====================================================")


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8000/ws/audio/?lang_in=es&lang_out=original"
    clients = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    sentences = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    asyncio.run(run_stress_test(url, clients, sentences))
