import logging
from starlette.types import ASGIApp, Scope, Receive, Send

logger = logging.getLogger(__name__)


class WSConnectionLoggingMiddleware:
    """
    ASGI Middleware to intercept and log WebSocket handshakes globally.
    """
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "websocket":
            client = scope.get("client")
            client_ip = client[0] if client else "unknown"
            path = scope.get("path", "")
            logger.info(f"[ASGI] WebSocket handshake from {client_ip} → '{path}'")

        await self.app(scope, receive, send)
