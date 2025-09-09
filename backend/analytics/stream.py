"""
Streaming metrics pipeline with backpressure and windowed aggregation.
"""
import asyncio
import time
from collections import defaultdict


class MetricsStream:
    def __init__(self, maxsize: int = 10_000):
        self.q = asyncio.Queue(maxsize=maxsize)

    async def publish(self, metric: dict) -> None:
        await self.q.put(metric)  # backpressure if full

    async def run(self, sink):
        buckets = defaultdict(list)
        last_flush = time.monotonic()
        while True:
            try:
                m = await asyncio.wait_for(self.q.get(), timeout=1.0)
                buckets[m["name"]].append((time.time(), m["value"]))
            except asyncio.TimeoutError:
                pass
            if time.monotonic() - last_flush > 5:
                for name, points in list(buckets.items()):
                    count = len(points)
                    if count == 0:
                        continue
                    avg = sum(v for _, v in points) / count
                    await sink.flush(name, avg, count, points[-1][0])
                buckets.clear()
                last_flush = time.monotonic()
