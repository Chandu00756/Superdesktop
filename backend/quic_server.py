"""
QUIC Control-Plane Server (scaffold)
- Uses aioquic to establish a control channel for mesh updates and probes.
"""
import asyncio
import logging
from typing import Optional

try:
    from aioquic.asyncio import serve
    from aioquic.asyncio.protocol import QuicConnectionProtocol
    from aioquic.quic.configuration import QuicConfiguration
    _HAS_AIOQUIC = True
except Exception:  # pragma: no cover - optional dep
    _HAS_AIOQUIC = False
    # Provide lightweight stubs to avoid NameError when the optional dependency isn't present
    class QuicConnectionProtocol:  # type: ignore
        pass

log = logging.getLogger(__name__)


class ControlProtocol(QuicConnectionProtocol):  # type: ignore[misc]
    async def quic_event_received(self, event):  # type: ignore[override]
        # Placeholder: handle events (STREAM_DATA, CONNECTION_TERMINATED, etc.)
        return None


async def run_quic(
    cert_path: str,
    key_path: str,
    host: str = "0.0.0.0",
    port: int = 4433,
    alpn: Optional[list[str]] = None,
) -> None:
    if not _HAS_AIOQUIC:
        # If aioquic isn't available, don't block startup; log once and return.
        log.warning("aioquic not installed; QUIC server disabled")
        return
    cfg = QuicConfiguration(is_client=False, alpn_protocols=alpn or ["hq-29", "h3"])
    cfg.load_cert_chain(cert_path, key_path)
    async with serve(host, port, configuration=cfg, create_protocol=ControlProtocol):
        log.info("QUIC server running on %s:%s", host, port)
        await asyncio.Future()
