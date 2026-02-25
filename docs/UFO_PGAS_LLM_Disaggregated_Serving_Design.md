# Design Doc: UFO-PGAS + LLM Disaggregated Serving

## Overview
This document outlines the architecture and implementation phases for integrating a **Partitioned Global Address Space (PGAS)** runtime—branded internally as **UFO-PGAS (Unifying Fabric Object)**—with **Disaggregated LLM Serving** capabilities in the Superdesktop environment.
The end state gives us a personal disaggregated rack OS that treats heterogeneous nodes as a single logical cluster for memory pooling, process migration, and LLM inference.

## 1. High-Level Architecture
The system fundamentally shifts from standard RPC to a PGAS-centric task model. The key pillars are:
1. **Memory Fabric (UFO-Memory)**: Manages local, CXL, and Ethernet-emulated distributed memory.
2. **Compute Fabric (UFO-Task)**: Dynamically places compute jobs (like LLM prefill vs. decode) based on data locality and hardware capability.
3. **Disaggregated Serving (UFO-LLM)**: Specifically orchestrates large language models using the custom memory and compute fabrics to share KV caches.

---

## 2. Core Modules

### 2.1 `memory_fabric` (UFO-PGAS)
Handles the global address space partitioned across cluster nodes.

- **`pgas_manager.py`**: The core controller allocating global virtual addresses (GVAs) to physical addresses (PAs). 
- **`cxl_allocator.py`**: A low-level tier allocator tracking page hotness. If CXL isn't present, it falls back to a software emulated RDMA/TCP cache.
- **`kv_cache_pool.py`**: An object-level zero-copy interface specifically tailored to storing layer-wise key/value tensors for inference operations.

### 2.2 `compute_fabric` (UFO-Task)
Graph-based runtime for application and LLM execution.

- **`task_graph.py`**: Represents incoming workloads as Directed Acyclic Graphs (DAGs) annotated with `ufo_in`, `ufo_out`, and `ufo_inout` dependencies.
- **`scheduler.py`**: Capability-aware scheduler that maps task execution. Variables tuned: GPU tensor cores, CPU vector widths, network latency, and memory locality.
- **`migrator.py`**: Facilitates suspend-and-resume mechanisms. It serializes thread/process state and rewrites PGAS references before resuming on a remote node.

### 2.3 `llm_serving` (UFO-LLM)
Specialized sub-module for disaggregated AI inference.

- **`prefill_worker.py`**: Designed to be scheduled on high-bandwidth, high-compute GPU nodes. Output goes directly to the `kv_cache_pool`.
- **`decode_worker.py`**: Designed for wide parallelism across multiple smaller devices. Reads from `kv_cache_pool` over the fabric.
- **`router.py`**: Decides how user prompts are split between prefill and decode stages.

---

## 3. Data Structures

### `UfoGlobalPointer`
A 64-bit struct representing an address in the PGAS.
- `node_id` (16 bits): Which node physically owns this page.
- `tier_id` (8 bits): Type of memory (0: Local DRAM, 1: CXL Pool, 2: Ethernet Emulation).
- `offset` (40 bits): Address offset within the node's partition.

### `TaskDescriptor`
Metadata for the UFO-Task scheduler.
```python
class TaskDescriptor:
    task_id: UUID
    required_capabilities: List[str]  # e.g., ["cuda", "fp16", "sm_80"]
    latency_sensitive: bool
    inputs: List[UfoGlobalPointer]
    outputs: List[UfoGlobalPointer]
    exec_state: bytes  # For migration
```

### `KVCacheBlock`
Object representing a layer of KV cache.
```python
class KVCacheBlock:
    block_id: int
    layer_idx: int
    token_range: Tuple[int, int]
    pgas_ptr: UfoGlobalPointer
    is_hot: bool
```

---

## 4. Phase-by-Phase Execution Plan

### Stage A: PGAS Prototyping & Emulation
**Goal**: Build a software-based PGAS backend.
1. **Develop `pgas_manager.py`**: Implement a global allocator mapping virtual addresses to logical nodes.
2. **Build Ethernet Emulation Layer**: Using fast TCP or gRPC (as a stand-in for RDMA) to fetch/store pages remotely.
3. **Experiment 1 (Ping-Pong Latency)**: Measure page-fault latency when accessing a `UfoGlobalPointer` mapped to a remote Raspberry Pi vs. local memory.

### Stage B: UFO-Task and Basic Migration
**Goal**: Move compute to where the memory is.
1. **Develop `task_graph.py` & `scheduler.py`**: Allow a Python script to define two tasks where Task B depends on Task A. 
2. **Experiment 2 (Compute Placement)**: Run Task A on Node 1 (generating 1GB of PGAS data), and observe the scheduler placing Task B on Node 1 to minimize cross-network data movement.
3. **Draft `migrator.py`**: Serialize a simple running Python generator, transfer the state to a new node, and resume execution.

### Stage C: Single-Model Disaggregated LLM Serving
**Goal**: Implement prefill/decode separation.
1. **Develop `kv_cache_pool.py`**: A dedicated API for Storing and Loading matrices from the PGAS.
2. **Split LLM Execution**: Modify an open-source inference script (e.g., `llama.cpp` or a PyTorch equivalent) to halt after prefill, write KVs into PGAS, and signal the `decode_worker`.
3. **Experiment 3 (Disaggregated Inference)**: Start the prefill phase on a high-end RTX GPU node. Execute decode sequentially on an edge node (e.g., Mac Mini or embedded device) retrieving KVs directly from the PGAS. Measure TTFT (Time To First Token) vs. Decode Tokens/Second against monolithic serving.

### Stage D: Hardware Disaggregation (CXL Integration)
**Goal**: Substitute Ethernet for native CXL pools.
1. **Implement `cxl_allocator.py`**: Use `mmap` or OS-native shared memory semantics directed to an actual CXL attached memory chassis.
2. **Implement Page Hotness Tracking**: Auto-promote highly accessed KV blocks from CXL pooled memory into local HBM/DRAM.
3. **Experiment 4 (Full Disaggregation Benchmark)**: Replicate Experiment 3, replacing the network fetching with standard memory load/store instructions over CXL. Compare memory bandwidth and tail latency directly against NVMe swap.

---

## Conclusion
By adopting the UFO-PGAS architecture, Superdesktop pivots from a conventional orchestrator to a true distributed operating system. The distinction between a local accelerator and a remote node dissolves, allowing the OS to arbitrarily mold around the workload at hand—reaching the ultimate goal of a homogeneous programming interface over a heavily heterogeneous cluster.
