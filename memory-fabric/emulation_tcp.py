"""
UFO-PGAS: TCP Emulation Backend
Provides the `TCPMemoryDevice` logic for fetching and storing pages across
network boundaries seamlessly, simulating CXL/RDMA memory disaggregation.
"""

import asyncio
import logging
from typing import Dict, Any, Awaitable, Callable

from pgas_manager import PGASDevice, UfoGlobalPointer

logger = logging.getLogger(__name__)

# Protocol Constants
CMD_FETCH = b"\x01"
CMD_STORE = b"\x02"
CMD_ALLOC = b"\x03"
CMD_DEALLOC = b"\x04"
CMD_ACK = b"\xFF"


class TCPMemoryDevice(PGASDevice):
    """
    Tier 2 Memory Device leveraging async TCP streams.
    Used for emulation of Disaggregated Shared Memory over Ethernet.
    """
    def __init__(self, device_id: str, tier_id: int, host: str, port: int, capacity_bytes: int):
        super().__init__(device_id, tier_id, capacity_bytes)
        self.host = host
        self.port = port
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None
        self.connect_lock = asyncio.Lock()
        
    async def _connect(self):
        """Establish connection to remote Memory Daemon."""
        async with self.connect_lock:
            if self.writer is None:
                logger.info(f"Connecting Disaggregated Memory Device {self.device_id} to {self.host}:{self.port}")
                self.reader, self.writer = await asyncio.open_connection(self.host, self.port)

    async def _send_request(self, cmd: bytes, payload: bytes) -> bytes:
        """Sends command and payload; reads the length-prefixed response."""
        if self.writer is None:
            await self._connect()
            
        try:
            # Format: [Cmd: 1B] [Payload Len: 8B] [Payload]
            header = cmd + len(payload).to_bytes(8, 'big')
            self.writer.write(header + payload)
            await self.writer.drain()
            
            # Read response length
            resp_len_bytes = await self.reader.readexactly(8)
            resp_len = int.from_bytes(resp_len_bytes, 'big')
            
            # Read response
            if resp_len > 0:
                return await self.reader.readexactly(resp_len)
            return b""
            
        except (ConnectionError, BrokenPipeError, asyncio.IncompleteReadError) as e:
            logger.error(f"Connection lost to {self.device_id}: {e}")
            self.writer = None
            raise RuntimeError(f"TCP Emulation layer failure: {e}")

    async def allocate(self, size_bytes: int) -> int:
        """Request allocation on the remote daemon."""
        payload = size_bytes.to_bytes(8, 'big')
        resp = await self._send_request(CMD_ALLOC, payload)
        
        # Expecting 8 byte offset returned
        if len(resp) != 8:
            raise RuntimeError(f"Allocation failed on {self.device_id}")
            
        offset = int.from_bytes(resp, 'big')
        self.allocated_bytes += size_bytes
        return offset

    async def deallocate(self, offset: int, size_bytes: int) -> None:
        """Request deallocation on the remote daemon."""
        payload = offset.to_bytes(8, 'big') + size_bytes.to_bytes(8, 'big')
        await self._send_request(CMD_DEALLOC, payload)
        self.allocated_bytes -= size_bytes

    async def fetch(self, offset: int, size_bytes: int) -> bytes:
        """Fetch memory block over network."""
        payload = offset.to_bytes(8, 'big') + size_bytes.to_bytes(8, 'big')
        data = await self._send_request(CMD_FETCH, payload)
        
        if len(data) != size_bytes:
            raise RuntimeError(f"Fetch mismatch: expected {size_bytes}, got {len(data)}")
        return data

    async def store(self, offset: int, data: bytes) -> None:
        """Store memory block over network."""
        payload = offset.to_bytes(8, 'big') + data
        resp = await self._send_request(CMD_STORE, payload)
        
        if resp != CMD_ACK:
            raise RuntimeError(f"Store failed on {self.device_id}")


class PGASTcpDaemon:
    """
    A standalone async TCP server that acts as a physical memory bank.
    It exposes the memory regions so `TCPMemoryDevice` clients can connect.
    """
    def __init__(self, host: str, port: int, capacity_bytes: int):
        self.host = host
        self.port = port
        
        # Using a bytearray as the physical memory region
        self.memory_region = bytearray(capacity_bytes)
        self.capacity_bytes = capacity_bytes
        self.next_offset = 0

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Processes incoming PGAS transport requests."""
        addr = writer.get_extra_info('peername')
        logger.debug(f"PGAS TCP connection accepted from {addr}")
        
        try:
            while True:
                # Read 9-byte header [cmd: 1B] [len: 8B]
                header = await reader.readexactly(9)
                cmd = header[0:1]
                payload_len = int.from_bytes(header[1:9], 'big')
                
                payload = await reader.readexactly(payload_len) if payload_len > 0 else b""
                response = b""
                
                if cmd == CMD_ALLOC:
                    size = int.from_bytes(payload[0:8], 'big')
                    if self.next_offset + size > self.capacity_bytes:
                        # Out of memory
                        response_len = b"\x00\x00\x00\x00\x00\x00\x00\x00" 
                        writer.write(response_len)
                        await writer.drain()
                        continue
                        
                    offset = self.next_offset
                    self.next_offset += size
                    response = offset.to_bytes(8, 'big')
                    
                elif cmd == CMD_DEALLOC:
                    # Soft de-allocation for prototype
                    response = CMD_ACK
                    
                elif cmd == CMD_FETCH:
                    offset = int.from_bytes(payload[0:8], 'big')
                    size = int.from_bytes(payload[8:16], 'big')
                    response = self.memory_region[offset:offset+size]
                    
                elif cmd == CMD_STORE:
                    offset = int.from_bytes(payload[0:8], 'big')
                    data = payload[8:]
                    
                    self.memory_region[offset:offset+len(data)] = data
                    response = CMD_ACK
                else:
                    logger.warning(f"Unknown command: {cmd}")
                    
                # Write back response
                writer.write(len(response).to_bytes(8, 'big') + response)
                await writer.drain()
                
        except asyncio.IncompleteReadError:
            logger.debug(f"PGAS TCP client {addr} disconnected expectedly.")
        except Exception as e:
            logger.error(f"PGAS TCP handler error for {addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def serve_forever(self):
        """Starts the physical daemon listener."""
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addrs = ', '.join(str(sockets.getsockname()) for sockets in server.sockets)
        logger.info(f"Serving PGAS daemon on {addrs} (Capacity: {self.capacity_bytes/(1024*1024):.2f} MB)")
        
        async with server:
            await server.serve_forever()
