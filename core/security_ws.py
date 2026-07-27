import os
import logging
from fastapi import WebSocket, WebSocketException, status, Query

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


async def verify_ws_client(
    websocket: WebSocket,
    token: str | None = Query(None, description="Authentication token for the OBS plugin")
):
    """
    Security/Authentication dependency for WebSockets.
    Executes BEFORE `websocket.accept()`.
    Reads OBS_API_KEY from environment variables (.env).
    """
    # Reads token from .env, fallback to "obs_secret_key" if not set
    valid_token = os.getenv("OBS_API_KEY", "obs_secret_key")

    if not token or token != valid_token:
        logger.warning(
            f"[WS Security] Connection rejected from {websocket.client.host}: "
            "Invalid or missing token."
        )
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Invalid or missing authentication token."
        )

    logger.info(f"[WS Security] Client verified from {websocket.client.host}")
    return token