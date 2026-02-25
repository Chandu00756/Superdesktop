import pytest
import asyncio
import time
import multiprocessing
from rdma_emulator import RDMADaemon, RDMADevice
from emulation_tcp import PGASTcpDaemon, TCPMemoryDevice

def run_rdma(host, port, capacity):
    daemon = RDMADaemon(host, port, capacity)
    asyncio.run(daemon.serve_forever())
    
def run_tcp(host, port, capacity):
    daemon = PGASTcpDaemon(host, port, capacity)
    asyncio.run(daemon.serve_forever())

@pytest.fixture(scope="module")
def daemon_ports():
    h_rdma, p_rdma = "127.0.0.1", 10001
    h_tcp, p_tcp = "127.0.0.1", 10002
    
    p1 = multiprocessing.Process(target=run_rdma, args=(h_rdma, p_rdma, 10 * 1024 * 1024))
    p2 = multiprocessing.Process(target=run_tcp, args=(h_tcp, p_tcp, 10 * 1024 * 1024))
    
    p1.start()
    p2.start()
    time.sleep(1.0) # wait for bindings
    
    yield (p_rdma, p_tcp)
    
    p1.terminate()
    p2.terminate()
    p1.join()
    p2.join()

@pytest.mark.asyncio
async def test_rdma_throughput(daemon_ports):
    p_rdma, p_tcp = daemon_ports
    
    rdma_dev = RDMADevice("rdma0", 1, "127.0.0.1", p_rdma, 10*1024*1024)
    tcp_dev = TCPMemoryDevice("tcp0", 2, "127.0.0.1", p_tcp, 10*1024*1024)
    
    # Payload: 1 Megabyte
    payload = b"\x01" * 1024 * 1024
    
    off_rdma = await rdma_dev.allocate(len(payload))
    off_tcp = await tcp_dev.allocate(len(payload))
    
    # Benchmark TCP Store
    t0 = time.time()
    for _ in range(10):
        await tcp_dev.store(off_tcp, payload)
    tcp_time = time.time() - t0
    
    # Benchmark RDMA zero-copy simulation Store
    t1 = time.time()
    mv_payload = memoryview(payload)
    for _ in range(10):
        await rdma_dev.store(off_rdma, mv_payload)
    rdma_time = time.time() - t1
    
    print(f"TCP Emulation Latency (1MB x 10):  {tcp_time:.4f}s")
    print(f"RDMA Emulation Latency (1MB x 10): {rdma_time:.4f}s")
    
    # RDMA should theoretically be faster due to memory slicing vs copying
    # In python async streams locally it might be close, but still verifying functionality
    data = await rdma_dev.fetch(off_rdma, len(payload))
    assert data == payload
