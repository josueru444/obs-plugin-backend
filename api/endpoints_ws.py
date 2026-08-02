import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import ValidationError

from core.security_ws import verify_ws_client
from core.whisper_ia import get_whisper_service
from core.audio_decoder import OpusAudioDecoder
from core.ai_translator import get_translator_service
from schemas.schemas import WSResponse, AudioQueryParams

logger = logging.getLogger(__name__)

router = APIRouter()

# WebSocket close codes
_WS_CLOSE_INTERNAL_ERROR = 1011
_WS_CLOSE_INVALID_DATA = 4003


@router.websocket('/ws/audio/')
async def websocket_endpoint(
        websocket: WebSocket,
        token: str = Depends(verify_ws_client),
):
    # ── Validate query params before accepting the connection ─────────────
    try:
        params = AudioQueryParams(
            lang_in=websocket.query_params.get("lang_in", "en"),
            lang_out=websocket.query_params.get("lang_out", "original"),
        )
    except ValidationError as exc:
        # Reject the connection with a descriptive reason before accepting
        await websocket.close(code=_WS_CLOSE_INVALID_DATA, reason=str(exc))
        logger.warning(f"[WS] Rejected connection: invalid language params — {exc}")
        return

    await websocket.accept()

    lang_in = params.lang_in
    lang_out = params.lang_out

    # Determine Whisper task and whether AI translation should take over.
    # If the translator is enabled, Whisper only transcribes (faster).
    # If the translator is disabled, Whisper translates natively (English only).
    translator = get_translator_service()
    ai_translation_active = (
        translator.is_enabled
        and lang_out not in ("original", lang_in)
    )

    if ai_translation_active:
        task = "transcribe"  # AI translator handles the translation
    elif lang_out == "en" and lang_in != "en":
        task = "translate"   # Whisper native translation (English only)
    else:
        task = "transcribe"

    whisper_svc = get_whisper_service()
    decoder = OpusAudioDecoder()

    # Tracks the most recent partial audio per sentence_id so we can
    # process the latest version when Whisper is finally free, instead
    # of a stale earlier version.
    latest_partial: dict[int, bytes] = {}

    logger.info(
        f"[WS] Client connected — lang_in={lang_in}, lang_out={lang_out}, "
        f"task={task}, ai_translation={ai_translation_active}"
    )

    try:
        while True:
            import asyncio
            data = await websocket.receive_bytes()

            # ── Drain queue to avoid backlog of stale partials ──────────────
            messages_to_process = [decoder.decode_message(data)]
            try:
                while True:
                    # Non-blocking peek (1ms timeout) to grab any queued packets
                    extra_data = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.001)
                    messages_to_process.append(decoder.decode_message(extra_data))
            except asyncio.TimeoutError:
                pass
            except Exception:
                pass

            # ── Collapse stale partials ─────────────────────────────────────
            # Keep all 'final' messages (so we don't lose completed sentences),
            # but only keep the VERY LAST message if it's a 'partial' (discarding older partials).
            finals = [m for m in messages_to_process if m.is_final]
            last_msg = messages_to_process[-1]
            
            if not last_msg.is_final:
                to_process = finals + [last_msg]
            else:
                to_process = finals

            for message in to_process:
                sid = message.sentence_id
            if ai_translation_active:
                # ── Pipeline: Whisper transcribes → AI translates (streaming) ──

                async def on_segment_with_ai(
                    text: str, is_final: bool, sid: int = sid
                ) -> None:
                    """
                    Streams each Whisper segment through the AI translator
                    and forwards tokens to the OBS plugin as they arrive.
                    The full accumulated translation is sent as a single
                    final message once streaming completes.
                    """
                    accumulated = ""
                    async for token in translator.translate_stream(text, lang_in, lang_out):
                        accumulated += token

                    # Send the full translated segment as a single response.
                    # We do NOT stream partial tokens because the LLM translates the whole
                    # growing sentence from scratch every second, which would cause a severe
                    # "re-typing" flicker in the OBS subtitles.
                    response = WSResponse(
                        text=accumulated.strip(),
                        sentence_id=sid,
                        is_final=is_final,
                    )
                    try:
                        await websocket.send_text(
                            response.model_dump_json(exclude_none=True)
                        )
                    except RuntimeError as exc:
                        if "websocket.close" in str(exc):
                            logger.info(f"[WS] Client disconnected before AI segment could be sent (sid={sid})")
                        else:
                            raise
                    except Exception as exc:
                        logger.warning(f"[WS] Error sending AI segment: {exc}")

                # 3. Run streaming transcription; on_segment_with_ai handles
                #    the AI translation for each resulting segment.
                await whisper_svc.transcribe_streaming(
                    audio_data=message.audio_pcm,
                    on_segment=on_segment_with_ai,
                    language=lang_in,
                    task=task,
                    is_final_message=message.is_final,
                )

            else:
                # ── Pipeline: Whisper transcribes/translates directly ──────────

                async def on_segment(
                    text: str, is_final: bool, sid: int = sid
                ) -> None:
                    response = WSResponse(
                        text=text,
                        sentence_id=sid,
                        is_final=is_final,
                    )
                    try:
                        await websocket.send_text(
                            response.model_dump_json(exclude_none=True)
                        )
                    except RuntimeError as exc:
                        if "websocket.close" in str(exc):
                            logger.info(f"[WS] Client disconnected before Whisper segment could be sent (sid={sid})")
                        else:
                            raise
                    except Exception as exc:
                        logger.warning(f"[WS] Error sending Whisper segment: {exc}")

                # 4. Run streaming transcription with the accumulated audio.
                await whisper_svc.transcribe_streaming(
                    audio_data=message.audio_pcm,
                    on_segment=on_segment,
                    language=lang_in,
                    task=task,
                    is_final_message=message.is_final,
                )

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected")
    except Exception as e:
        logger.error(f"[WS] Unhandled error: {e}", exc_info=True)
        try:
            await websocket.close(code=_WS_CLOSE_INTERNAL_ERROR)
        except Exception:
            pass  # Connection may already be closed