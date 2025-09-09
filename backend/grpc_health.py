"""
gRPC Health Service for Omega
- Exposes standard grpc.health.v1.Health with async server
- Integrates with an asyncio.Event to reflect readiness
"""
import asyncio
import logging
from typing import Optional

import grpc
from grpc_health.v1 import health_pb2_grpc, health_pb2

log = logging.getLogger(__name__)


class HealthServicer(health_pb2_grpc.HealthServicer):  # type: ignore[misc]
    def __init__(self, ready_event: asyncio.Event):
        super().__init__()
        self._ready = ready_event

    async def Check(self, request, context):  # type: ignore[override]
        status = (
            health_pb2.HealthCheckResponse.SERVING
            if self._ready.is_set()
            else health_pb2.HealthCheckResponse.NOT_SERVING
        )
        return health_pb2.HealthCheckResponse(status=status)

    async def Watch(self, request, context):  # type: ignore[override]
        # Simple watch implementation that streams current status and then completes
        status = (
            health_pb2.HealthCheckResponse.SERVING
            if self._ready.is_set()
            else health_pb2.HealthCheckResponse.NOT_SERVING
        )
        await context.write(health_pb2.HealthCheckResponse(status=status))


async def serve_grpc_health(
    ready_event: asyncio.Event,
    host: str = "0.0.0.0",
    port: int = 50051,
    max_msg_mb: int = 50,
) -> None:
    """Start an async gRPC server exposing the health service.

    This coroutine runs until cancelled. Call it within a background task.
    """
    options = [
        ("grpc.max_send_message_length", max_msg_mb * 1024 * 1024),
        ("grpc.max_receive_message_length", max_msg_mb * 1024 * 1024),
    ]
    server = grpc.aio.server(options=options)
    svc = HealthServicer(ready_event)
    health_pb2_grpc.add_HealthServicer_to_server(svc, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    log.info("gRPC health server started on %s:%s", host, port)
    try:
        await server.wait_for_termination()
    except asyncio.CancelledError:
        log.info("gRPC health server cancellation received")
        await server.stop(0)
        raise
