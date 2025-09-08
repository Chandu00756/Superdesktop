"""Streaming negotiation scaffold.

Provides minimal in-memory WebRTC-like offer/answer exchange so the
desktop can establish peer transports later (actual media pipeline not
implemented yet). Each stream entry persists for a short TTL and is
addressable by a ULID-like identifier. Designed to be upgraded to true
WebRTC/QUIC handlers with TURN/STUN allocation.

Events:
  stream.offer.created
  stream.answer.attached
  stream.stream.expired

Security: Only metadata (SDP blobs) stored transiently in-process; no
media frames handled here.
"""
from __future__ import annotations
import asyncio, time, uuid
from typing import Dict, Any, Optional

try:
    from common.event_bus import get_event_bus  # type: ignore
except Exception:  # pragma: no cover - fallback if import path issues
    async def get_event_bus():  # type: ignore
        class _NullBus:
            async def publish(self, *a, **k):
                return ''
        return _NullBus()


class StreamStore:
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self._streams: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self):
        if self._started:
            return
        self._started = True
        asyncio.create_task(self._janitor())

    async def _janitor(self):
        while True:
            await asyncio.sleep(30)
            now = time.time()
            expired = [sid for sid, meta in self._streams.items() if now - meta['created'] > self.ttl]
            if not expired:
                continue
            bus = await get_event_bus()
            async with self._lock:
                for sid in expired:
                    self._streams.pop(sid, None)
                    try:
                        await bus.publish('stream', 'stream.stream.expired', {'stream_id': sid})
                    except Exception:
                        pass

    async def create_offer(self, session_id: str, sdp: str) -> str:
        sid = uuid.uuid4().hex
        rec = {'stream_id': sid, 'session_id': session_id, 'offer_sdp': sdp, 'answer_sdp': None, 'created': time.time()}
        async with self._lock:
            self._streams[sid] = rec
        bus = await get_event_bus()
        try:
            await bus.publish('stream', 'stream.offer.created', {'stream_id': sid, 'session_id': session_id})
        except Exception:
            pass
        return sid

    async def attach_answer(self, stream_id: str, sdp: str) -> bool:
        async with self._lock:
            rec = self._streams.get(stream_id)
            if not rec:
                return False
            rec['answer_sdp'] = sdp
        bus = await get_event_bus()
        try:
            await bus.publish('stream', 'stream.answer.attached', {'stream_id': stream_id})
        except Exception:
            pass
        return True

    async def get(self, stream_id: str) -> Optional[Dict[str, Any]]:
        return self._streams.get(stream_id)

    async def delete(self, stream_id: str) -> bool:
        async with self._lock:
            existed = stream_id in self._streams
            self._streams.pop(stream_id, None)
        return existed


# Global singleton
_store: Optional[StreamStore] = None

async def get_stream_store() -> StreamStore:
    global _store
    if _store is None:
        _store = StreamStore()
        await _store.start()
    return _store

__all__ = ['get_stream_store', 'StreamStore']
