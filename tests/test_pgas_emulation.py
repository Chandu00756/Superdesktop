import pytest
import asyncio
import multiprocessing
import time
from emulation_tcp import PGASTcpDaemon, TCPMemoryDevice

def run_daemon(host, port, capacity):
    daemon = PGASTcpDaemon(host, port, capacity)
    asyncio.run(daemon.serve_forever())

@pytest.fixture(scope="module")
def pgas_daemon():
    host, port = "127.0.0.1", 9999
    # Start the daemon in a separate process
    p = multiprocessing.Process(target=run_daemon, args=(host, port, 1024*1024))
    p.start()
    time.sleep(0.5) # Give it time to bind
    
    yield (host, port)
    
    p.terminate()
    p.join()

@pytest.mark.asyncio
async def test_tcp_emulation_device(pgas_daemon):
    host, port = pgas_daemon
    
    device = TCPMemoryDevice("remote0", tier_id=2, host=host, port=port, capacity_bytes=1024*1024)
    
    # 1. Allocate 1KB
    offset = await device.allocate(1024)
    assert offset == 0
    
    # 2. Store Data
    payload = b"UFO-PGAS Network Payload!"
    await device.store(offset, payload)
    
    # 3. Fetch Data
    fetched = await device.fetch(offset, len(payload))
    assert fetched == payload
    
    # 4. Deallocate
    await device.deallocate(offset, 1024)
    assert device.allocated_bytes == 0
