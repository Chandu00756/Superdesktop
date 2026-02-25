"""
UFO-PGAS: RDMA Emulator (Zero-Copy)
A lower-latency transport layer simulating Remote Direct Memory Access (RDMA).
Uses `memoryview` and shared byte arrays to avoid serialization overhead
common in standard TCP applications.
"""

import asyncio
import logging
import struct
from typing import Optional

from pgas_manager import PGASDevice

logger = logging.getLogger(__name__)

# Protocol Constants
CMD_FETCH = b"\x01"
CMD_STORE = b"\x02"
CMD_ALLOC = b"\x03"
CMD_DEALLOC = b"\x04"
CMD_ACK = b"\xFF"


class RDMADevice(PGASDevice):
    """
    Tier 2 Memory Device leveraging async streams with `memoryview` 
    transfers to reduce CPU footprint and copy overhead.
    """
    def __init__(self, device_id: str, tier_id: int, host: str, port: int, capacity_bytes: int):
        super().__init__(device_id, tier_id, capacity_bytes)
        self.host = host
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.connect_lock = asyncio.Lock()
        
    async def _connect(self):
        async with self.connect_lock:
            if self.writer is None:
                logger.info(f"Connecting RDMA Fabric Emulator to {self.host}:{self.port}")
                self.reader, self.writer = await asyncio.open_connection(self.host, self.port)

    async def _send_request(self, cmd: bytes, payload: memoryview | bytes) -> bytes:
        if self.writer is None:
            await self._connect()
            
        try:
            # Format: [Cmd: 1B] [Payload Len: 8B]
            payload_len = len(payload)
            header = cmd + struct.pack("!Q", payload_len)
            
            # Write header directly
            self.writer.write(header)
            
            # Write payload directly from memory buffer (zero-copy attempt via TCP transport)
            self.writer.write(payload)
            await self.writer.drain()
            
            # Read response
            resp_len_bytes = await self.reader.readexactly(8)
            resp_len = struct.unpack("!Q", resp_len_bytes)[0]
            
            if resp_len > 0:
                data = await self.reader.readexactly(resp_len)
                return data
            return b""
            
        except Exception as e:
            logger.error(f"RDMA Emulator connection lost: {e}")
            self.writer = None
            raise

    async def allocate(self, size_bytes: int) -> int:
        payload = struct.pack("!Q", size_bytes)
        resp = await self._send_request(CMD_ALLOC, payload)
        if len(resp) != 8:
            raise RuntimeError(f"Allocation failed on {self.device_id}")
        offset = struct.unpack("!Q", resp)[0]
        self.allocated_bytes += size_bytes
        return offset

    async def deallocate(self, offset: int, size_bytes: int) -> None:
        payload = struct.pack("!QQ", offset, size_bytes)
        await self._send_request(CMD_DEALLOC, payload)
        self.allocated_bytes -= size_bytes

    async def fetch(self, offset: int, size_bytes: int) -> bytes:
        payload = struct.pack("!QQ", offset, size_bytes)
        data = await self._send_request(CMD_FETCH, payload)
        if len(data) != size_bytes:
            raise RuntimeError(f"Fetch mismatch: expected {size_bytes}, got {len(data)}")
        return data

    async def store(self, offset: int, data: bytes | bytearray | memoryview) -> None:
        """
        Takes raw byte blocks and sends them over the socket.
        """
        # Prefix with 8-byte offset
        offset_bytes = struct.pack("!Q", offset)
        
        # We can construct a zero-copy send list if supported, but for asyncio streams
        # we concatenate. In true RDMA, this bypasses the kernel completely.
        # Still faster than old implementation due to struct packing vs int.to_bytes
        if isinstance(data, memoryview):
            payload = offset_bytes + data.tobytes()
        else:
            payload = offset_bytes + data
            
        resp = await self._send_request(CMD_STORE, payload)
        if resp != CMD_ACK:
            raise RuntimeError(f"Store failed on {self.device_id}")


class RDMADaemon:
    """
    Physical backend listener for the RDMADevice. Uses pre-allocated bytearray memory regions.
    """
    def __init__(self, host: str, port: int, capacity_bytes: int):
        self.host = host
        self.port = port
        self.memory_region = bytearray(capacity_bytes)
        self.capacity_bytes = capacity_bytes
        self.next_offset = 0

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            while True:
                header = await reader.readexactly(9)
                cmd = header[0:1]
                payload_len = struct.unpack("!Q", header[1:9])[0]
                
                payload = await reader.readexactly(payload_len) if payload_len > 0 else b""
                response = b""
                
                if cmd == CMD_ALLOC:
                    size = struct.unpack("!Q", payload)[0]
                    if self.next_offset + size > self.capacity_bytes:
                        response_len = b"\x00\x00\x00\x00\x00\x00\x00\x00"
                        writer.write(response_len)
                        await writer.drain()
                        continue
                        
                    offset = self.next_offset
                    self.next_offset += size
                    response = struct.pack("!Q", offset)
                    
                elif cmd == CMD_DEALLOC:
                    response = CMD_ACK
                    
                elif cmd == CMD_FETCH:
                    offset, size = struct.unpack("!QQ", payload)
                    # Memoryview slicing prevents copy on read from the buffer
                    response = memoryview(self.memory_region)[offset:offset+size]
                    
                elif cmd == CMD_STORE:
                    offset = struct.unpack("!Q", payload[:8])[0]
                    data_view = memoryview(payload)[8:]
                    
                    # Direct copy into pre-allocated memory bytes
                    self.memory_region[offset:offset+len(data_view)] = data_view
                    response = CMD_ACK
                    
                # Write back response length and data
                writer.write(struct.pack("!Q", len(response)) + response)
                await writer.drain()
                
        except asyncio.IncompleteReadError:
            pass
        except Exception as e:
            logger.error(f"RDMA Daemon Error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def serve_forever(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        await server.serve_forever()
