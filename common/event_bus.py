"""Asynchronous Event Bus Abstraction.

Primary implementation uses Redis Streams (XADD / XREAD) if redis-py is available
and REDIS_URL is configured. Falls back to in-process asyncio.Queue channels so
local development works without infrastructure.

Features:
- Publish / subscribe named channels (logical topics)
- Consumer group support stub (future extension)
- Backpressure handling (bounded queue)
- JSON payload normalization + envelope (id, ts, type, data)

The interface purposely minimal to allow swapping to NATS or Kafka later.
"""
from __future__ import annotations
import os, json, time, asyncio, logging, uuid
from typing import Any, Dict, AsyncIterator, Optional

log = logging.getLogger(__name__)

class EventBus:
    def __init__(self, channel_prefix: str = "omega", memory_queue_size: int = 1000):
        self.prefix = channel_prefix
        self._queues: Dict[str, asyncio.Queue] = {}
        self._redis = None
        self._lock = asyncio.Lock()
        self._memory_queue_size = memory_queue_size

    async def initialize(self):
        if self._redis:  # already init
            return
        url = os.getenv('REDIS_URL') or os.getenv('OMEGA_REDIS_URL')
        if not url:
            log.debug("EventBus: no REDIS_URL set, using in-memory queues")
            return
        try:
            import redis.asyncio as redis  # type: ignore
            self._redis = redis.from_url(url, decode_responses=True)
            await self._redis.ping()
            log.info("EventBus connected to Redis")
        except Exception as e:
            log.warning(f"EventBus: Redis unavailable ({e}); falling back to memory queues")
            self._redis = None

    def _chan(self, name: str) -> str:
        return f"{self.prefix}:{name}" if not name.startswith(f"{self.prefix}:") else name

    async def publish(self, channel: str, event_type: str, data: Dict[str, Any]) -> str:
        eid = uuid.uuid4().hex
        envelope = {
            'id': eid,
            'ts': time.time(),
            'type': event_type,
            'data': data
        }
        chan = self._chan(channel)
        if self._redis:
            try:
                # Use Redis Streams (approx ordering per channel)
                await self._redis.xadd(chan, { 'event': json.dumps(envelope, separators=(',',':')) })
                return eid
            except Exception as e:
                log.debug(f"EventBus publish redis error {e}; fallback to memory for {chan}")
        # memory fallback
        q = self._queues.get(chan)
        if not q:
            q = asyncio.Queue(maxsize=self._memory_queue_size)
            self._queues[chan] = q
        try:
            q.put_nowait(envelope)
        except asyncio.QueueFull:
            # drop oldest strategy: drain one then put
            try:
                _ = q.get_nowait()
            except Exception:
                pass
            q.put_nowait(envelope)
        return eid

    async def subscribe(self, channel: str, last_id: str = "0-0") -> AsyncIterator[Dict[str, Any]]:
        chan = self._chan(channel)
        # Redis stream tailing
        if self._redis:
            while True:
                try:
                    resp = await self._redis.xread({chan: last_id}, block=5000, count=100)
                    if not resp:
                        continue
                    for (_key, entries) in resp:
                        for sid, fields in entries:
                            raw = fields.get('event')
                            if raw:
                                try:
                                    evt = json.loads(raw)
                                    last_id = sid
                                    yield evt
                                except Exception:
                                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.debug(f"EventBus subscribe redis error: {e}; switching to memory queue")
                    break
        # Memory queue consumer
        q = self._queues.get(chan)
        if not q:
            q = asyncio.Queue()
            self._queues[chan] = q
        while True:
            try:
                evt = await q.get()
                yield evt
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug(f"EventBus memory subscribe error: {e}")
                await asyncio.sleep(0.5)

# Global factory (lazy)
_global_bus: Optional[EventBus] = None
async def get_event_bus() -> EventBus:
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
        await _global_bus.initialize()
    return _global_bus

__all__ = ['EventBus','get_event_bus']
