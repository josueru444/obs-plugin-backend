import numpy as np
import logging
from core.whisper_ia import get_whisper_service

logging.basicConfig(level=logging.INFO)

svc = get_whisper_service()
audio = np.zeros(16000 * 2, dtype=np.float32) # 2 seconds of silence
import asyncio

async def main():
    async def on_segment(text, is_final):
        pass
    
    print("Testing translation task")
    await svc.transcribe_streaming(audio, on_segment, language="es", task="translate", is_final_message=True)

asyncio.run(main())
