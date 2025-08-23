"""
Omega Super Desktop Console - Render Router Service
Advanced Graphics Rendering Routing and Optimization Service

This service manages intelligent routing of rendering workloads across distributed
GPU resources, implements advanced encoding optimization, and provides real-time
frame rate optimization for ultra-low latency desktop streaming.

Key Features:
- Intelligent GPU workload distribution
- Real-time encoding optimization (H.264, H.265, AV1)
- Frame rate adaptation and motion compensation
- HDR and wide color gamut support
- Variable refresh rate (VRR) optimization
- Multi-stream rendering for multiple displays
"""

import asyncio
import json
import logging
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import os
import threading
import queue
import numpy as np

# Core Dependencies
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import aioredis
import asyncpg
import grpc
from grpc import aio as aio_grpc

# GPU and Graphics
try:
    import py3nvml.py3nvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("NVML not available - GPU monitoring disabled")

# Image and Video Processing
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available - some features disabled")

# Monitoring
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from utils.metrics import create_counter, create_gauge, create_histogram

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus Metrics
render_requests_total = create_counter('render_requests_total', 'Total render requests', ['gpu_id', 'encoding'])
frame_render_duration = create_histogram('frame_render_duration_ms', 'Frame render time in milliseconds')
encoding_duration = create_histogram('encoding_duration_ms', 'Frame encoding time in milliseconds')
frame_drops_total = create_counter('frame_drops_total', 'Total dropped frames', ['reason'])
bandwidth_utilization = create_gauge('bandwidth_utilization_mbps', 'Current bandwidth utilization')
gpu_render_utilization = create_gauge('gpu_render_utilization_percent', 'GPU render utilization', ['gpu_id'])
active_streams = create_gauge('active_render_streams', 'Number of active render streams')

# Rendering and Encoding Types
class RenderingAPI(Enum):
    DIRECTX12 = "directx12"
    VULKAN = "vulkan"
    OPENGL = "opengl"
    METAL = "metal"
    CUDA = "cuda"
    OPENCL = "opencl"

class EncodingCodec(Enum):
    H264 = "h264"
    H265 = "h265"
    AV1 = "av1"
    VP9 = "vp9"
    MJPEG = "mjpeg"

class EncodingPreset(Enum):
    ULTRAFAST = "ultrafast"
    SUPERFAST = "superfast"  
    VERYFAST = "veryfast"
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    SLOWER = "slower"
    VERYSLOW = "veryslow"

class ColorSpace(Enum):
    REC709 = "rec709"
    REC2020 = "rec2020"
    DCI_P3 = "dci_p3"
    ADOBE_RGB = "adobe_rgb"

@dataclass
class DisplayConfiguration:
    """Display configuration for rendering"""
    display_id: str
    resolution_width: int
    resolution_height: int
    refresh_rate: float
    color_depth: int
    color_space: ColorSpace
    hdr_enabled: bool
    variable_refresh_rate: bool
    adaptive_sync: bool
    primary_display: bool
    
    @property
    def pixel_count(self) -> int:
        return self.resolution_width * self.resolution_height
    
    @property
    def aspect_ratio(self) -> float:
        return self.resolution_width / self.resolution_height

@dataclass
class RenderingCapabilities:
    """GPU rendering capabilities"""
    gpu_id: str
    node_id: str
    device_name: str
    compute_capability: str
    memory_total_mb: int
    memory_available_mb: int
    
    # Rendering Features
    max_texture_size: int
    max_render_targets: int
    supports_raytracing: bool
    supports_mesh_shaders: bool
    supports_variable_rate_shading: bool
    
    # Encoding Features
    hardware_encoders: List[EncodingCodec]
    max_encoding_sessions: int
    supports_b_frames: bool
    supports_lookahead: bool
    
    # Performance Characteristics
    base_clock_mhz: int
    boost_clock_mhz: int
    memory_bandwidth_gbps: float
    render_output_units: int
    streaming_multiprocessors: int
    
    # Current State
    current_utilization_percent: float
    current_temperature_celsius: int
    current_power_draw_watts: int

@dataclass
class StreamConfiguration:
    """Stream configuration for client"""
    stream_id: str
    session_id: str
    client_id: str
    
    # Video Configuration
    target_resolution: Tuple[int, int]
    target_fps: int
    encoding_codec: EncodingCodec
    encoding_preset: EncodingPreset
    bitrate_mbps: float
    keyframe_interval: int
    
    # Quality Settings
    quality_level: float  # 0.0 to 1.0
    adaptive_quality: bool
    low_latency_mode: bool
    
    # Client Capabilities
    client_capabilities: Dict[str, Any]
    network_bandwidth_mbps: float
    network_latency_ms: float
    
    # Display Configuration
    displays: List[DisplayConfiguration]
    
    # Advanced Features
    hdr_tone_mapping: bool
    motion_vectors: bool
    frame_pacing: bool
    vsync_enabled: bool

@dataclass
class RenderTask:
    """Individual render task"""
    task_id: str
    stream_id: str
    frame_number: int
    timestamp: datetime
    
    # Render Parameters
    viewport: Tuple[int, int, int, int]  # x, y, width, height
    rendering_api: RenderingAPI
    render_targets: List[str]
    
    # Performance Requirements
    max_render_time_ms: float
    priority_level: int
    
    # Application Context
    application_id: str
    window_handle: Optional[str]
    gpu_context: Optional[str]

@dataclass
class FrameMetrics:
    """Per-frame performance metrics"""
    frame_number: int
    stream_id: str
    timestamp: datetime
    
    # Timing Metrics
    render_start_time: float
    render_end_time: float
    encode_start_time: float
    encode_end_time: float
    transmission_start_time: float
    
    # Quality Metrics
    frame_size_bytes: int
    quality_score: float
    motion_intensity: float
    complexity_score: float
    
    # Performance Metrics
    gpu_utilization_during_render: float
    memory_usage_mb: int
    dropped: bool
    drop_reason: Optional[str]

class GPUResourceManager:
    """Manages GPU resources and workload distribution"""
    
    def __init__(self):
        self.gpu_capabilities: Dict[str, RenderingCapabilities] = {}
        self.active_tasks: Dict[str, RenderTask] = {}
        self.gpu_queues: Dict[str, asyncio.Queue] = {}
        self.load_balancer_weights: Dict[str, float] = {}
        
    async def initialize(self):
        """Initialize GPU resource discovery"""
        if not NVML_AVAILABLE:
            logger.warning("NVML not available - using mock GPU data")
            await self._create_mock_gpus()
            return
        
        try:
            nvml.nvmlInit()
            gpu_count = nvml.nvmlDeviceGetCount()
            
            for i in range(gpu_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                capabilities = await self._discover_gpu_capabilities(handle, i)
                
                self.gpu_capabilities[capabilities.gpu_id] = capabilities
                self.gpu_queues[capabilities.gpu_id] = asyncio.Queue(maxsize=100)
                self.load_balancer_weights[capabilities.gpu_id] = 1.0
                
                logger.info(
                    "GPU discovered",
                    gpu_id=capabilities.gpu_id,
                    device_name=capabilities.device_name,
                    memory_mb=capabilities.memory_total_mb
                )
            
        except Exception as e:
            logger.error("Failed to initialize GPU resources", error=str(e))
            await self._create_mock_gpus()
    
    async def _create_mock_gpus(self):
        """Create mock GPU data for testing"""
        for i in range(2):
            gpu_id = f"gpu-{i}"
            capabilities = RenderingCapabilities(
                gpu_id=gpu_id,
                node_id=os.getenv('NODE_ID', 'localhost'),
                device_name=f"Mock RTX 4090 {i}",
                compute_capability="8.9",
                memory_total_mb=24576,
                memory_available_mb=20480,
                max_texture_size=32768,
                max_render_targets=8,
                supports_raytracing=True,
                supports_mesh_shaders=True,
                supports_variable_rate_shading=True,
                hardware_encoders=[EncodingCodec.H264, EncodingCodec.H265, EncodingCodec.AV1],
                max_encoding_sessions=4,
                supports_b_frames=True,
                supports_lookahead=True,
                base_clock_mhz=2200,
                boost_clock_mhz=2520,
                memory_bandwidth_gbps=1008.0,
                render_output_units=176,
                streaming_multiprocessors=128,
                current_utilization_percent=np.random.uniform(20, 60),
                current_temperature_celsius=np.random.randint(45, 75),
                current_power_draw_watts=np.random.randint(200, 400)
            )
            
            self.gpu_capabilities[gpu_id] = capabilities
            self.gpu_queues[gpu_id] = asyncio.Queue(maxsize=100)
            self.load_balancer_weights[gpu_id] = 1.0
    
    async def _discover_gpu_capabilities(self, handle, index: int) -> RenderingCapabilities:
        """Discover GPU capabilities using NVML"""
        device_name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
        uuid = nvml.nvmlDeviceGetUUID(handle).decode('utf-8')
        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
        
        try:
            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
            current_utilization = utilization.gpu
        except:
            current_utilization = 0
        
        try:
            temperature = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
        except:
            temperature = 60
        
        try:
            power_draw = nvml.nvmlDeviceGetPowerUsage(handle) // 1000
        except:
            power_draw = 250
        
        return RenderingCapabilities(
            gpu_id=uuid,
            node_id=os.getenv('NODE_ID', 'localhost'),
            device_name=device_name,
            compute_capability="8.6",  # Would query actual capability
            memory_total_mb=memory_info.total // (1024 * 1024),
            memory_available_mb=memory_info.free // (1024 * 1024),
            max_texture_size=32768,
            max_render_targets=8,
            supports_raytracing=True,
            supports_mesh_shaders=True,
            supports_variable_rate_shading=True,
            hardware_encoders=[EncodingCodec.H264, EncodingCodec.H265, EncodingCodec.AV1],
            max_encoding_sessions=4,
            supports_b_frames=True,
            supports_lookahead=True,
            base_clock_mhz=1700,
            boost_clock_mhz=1900,
            memory_bandwidth_gbps=900.0,
            render_output_units=112,
            streaming_multiprocessors=84,
            current_utilization_percent=current_utilization,
            current_temperature_celsius=temperature,
            current_power_draw_watts=power_draw
        )
    
    async def select_optimal_gpu(self, render_task: RenderTask) -> Optional[str]:
        """Select optimal GPU for rendering task"""
        if not self.gpu_capabilities:
            return None
        
        best_gpu = None
        best_score = float('-inf')
        
        for gpu_id, capabilities in self.gpu_capabilities.items():
            # Check if GPU can handle the task
            if not await self._can_handle_task(gpu_id, render_task):
                continue
            
            # Calculate suitability score
            score = await self._calculate_gpu_score(gpu_id, render_task)
            
            if score > best_score:
                best_score = score
                best_gpu = gpu_id
        
        return best_gpu
    
    async def _can_handle_task(self, gpu_id: str, task: RenderTask) -> bool:
        """Check if GPU can handle the rendering task"""
        capabilities = self.gpu_capabilities[gpu_id]
        
        # Check memory availability
        estimated_memory_mb = self._estimate_memory_usage(task)
        if estimated_memory_mb > capabilities.memory_available_mb:
            return False
        
        # Check current load
        queue_size = self.gpu_queues[gpu_id].qsize()
        if queue_size > 50:  # Queue too full
            return False
        
        return True
    
    def _estimate_memory_usage(self, task: RenderTask) -> int:
        """Estimate memory usage for rendering task"""
        viewport = task.viewport
        width = viewport[2]
        height = viewport[3]
        
        # Rough estimation: 4 bytes per pixel * 2 (front/back buffer) + overhead
        base_memory = (width * height * 4 * 2) // (1024 * 1024)  # MB
        overhead = 100  # MB for shaders, textures, etc.
        
        return base_memory + overhead
    
    async def _calculate_gpu_score(self, gpu_id: str, task: RenderTask) -> float:
        """Calculate suitability score for GPU"""
        capabilities = self.gpu_capabilities[gpu_id]
        
        # Base performance score (40% weight)
        performance_score = (
            capabilities.boost_clock_mhz / 3000.0 * 0.4 +
            capabilities.memory_bandwidth_gbps / 1000.0 * 0.3 +
            capabilities.streaming_multiprocessors / 128.0 * 0.3
        )
        
        # Current load score (40% weight) - prefer less loaded GPUs
        current_load = capabilities.current_utilization_percent / 100.0
        load_score = max(0.0, 1.0 - current_load)
        
        # Memory availability score (20% weight)
        memory_ratio = capabilities.memory_available_mb / capabilities.memory_total_mb
        memory_score = memory_ratio
        
        total_score = performance_score * 0.4 + load_score * 0.4 + memory_score * 0.2
        
        return total_score
    
    async def update_gpu_metrics(self):
        """Update GPU utilization and availability metrics"""
        for gpu_id, capabilities in self.gpu_capabilities.items():
            # Update Prometheus metrics
            gpu_render_utilization.labels(gpu_id=gpu_id).set(
                capabilities.current_utilization_percent
            )
            
            # Update capabilities with current data (mock for now)
            capabilities.current_utilization_percent = np.random.uniform(20, 80)
            capabilities.current_temperature_celsius = np.random.randint(45, 80)
            capabilities.current_power_draw_watts = np.random.randint(200, 450)

class EncodingOptimizer:
    """Optimizes encoding parameters for best quality/performance balance"""
    
    def __init__(self):
        self.encoding_profiles: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[Dict]] = {}
        
        # Initialize encoding profiles
        self._initialize_encoding_profiles()
    
    def _initialize_encoding_profiles(self):
        """Initialize encoding profiles for different scenarios"""
        self.encoding_profiles = {
            "ultra_low_latency": {
                "codec": EncodingCodec.H264,
                "preset": EncodingPreset.ULTRAFAST,
                "keyframe_interval": 1,
                "b_frames": 0,
                "lookahead": 0,
                "rate_control": "cbr",
                "quality_target": 0.7
            },
            "low_latency": {
                "codec": EncodingCodec.H264,
                "preset": EncodingPreset.FAST,
                "keyframe_interval": 2,
                "b_frames": 1,
                "lookahead": 10,
                "rate_control": "vbr",
                "quality_target": 0.8
            },
            "balanced": {
                "codec": EncodingCodec.H265,
                "preset": EncodingPreset.MEDIUM,
                "keyframe_interval": 4,
                "b_frames": 2,
                "lookahead": 20,
                "rate_control": "vbr",
                "quality_target": 0.85
            },
            "high_quality": {
                "codec": EncodingCodec.H265,
                "preset": EncodingPreset.SLOW,
                "keyframe_interval": 8,
                "b_frames": 4,
                "lookahead": 40,
                "rate_control": "vbr",
                "quality_target": 0.9
            },
            "bandwidth_optimized": {
                "codec": EncodingCodec.AV1,
                "preset": EncodingPreset.MEDIUM,
                "keyframe_interval": 6,
                "b_frames": 3,
                "lookahead": 30,
                "rate_control": "vbr",
                "quality_target": 0.8
            }
        }
    
    def select_optimal_profile(self, stream_config: StreamConfiguration) -> Dict[str, Any]:
        """Select optimal encoding profile based on stream requirements"""
        
        # Determine profile based on requirements
        if stream_config.low_latency_mode and stream_config.target_fps >= 120:
            profile_name = "ultra_low_latency"
        elif stream_config.low_latency_mode:
            profile_name = "low_latency"
        elif stream_config.network_bandwidth_mbps < 50:
            profile_name = "bandwidth_optimized"
        elif stream_config.quality_level >= 0.9:
            profile_name = "high_quality"
        else:
            profile_name = "balanced"
        
        base_profile = self.encoding_profiles[profile_name].copy()
        
        # Adapt based on stream configuration
        base_profile = self._adapt_profile_to_stream(base_profile, stream_config)
        
        return base_profile
    
    def _adapt_profile_to_stream(self, profile: Dict[str, Any], stream_config: StreamConfiguration) -> Dict[str, Any]:
        """Adapt encoding profile to specific stream requirements"""
        
        # Adjust bitrate based on resolution and quality
        resolution_factor = (stream_config.target_resolution[0] * stream_config.target_resolution[1]) / (1920 * 1080)
        quality_factor = stream_config.quality_level
        fps_factor = stream_config.target_fps / 60.0
        
        base_bitrate = 10.0  # Mbps for 1080p60
        target_bitrate = base_bitrate * resolution_factor * quality_factor * fps_factor
        
        # Clamp to available bandwidth
        target_bitrate = min(target_bitrate, stream_config.network_bandwidth_mbps * 0.8)
        
        profile["bitrate_mbps"] = target_bitrate
        
        # Adjust keyframe interval based on FPS
        if stream_config.target_fps >= 120:
            profile["keyframe_interval"] = max(1, profile["keyframe_interval"] // 2)
        
        # Disable B-frames for ultra-low latency
        if stream_config.low_latency_mode and stream_config.network_latency_ms < 10:
            profile["b_frames"] = 0
            profile["lookahead"] = 0
        
        return profile

class RenderRouter:
    """Main render routing service"""
    
    def __init__(self):
        self.gpu_manager = GPUResourceManager()
        self.encoding_optimizer = EncodingOptimizer()
        self.active_streams: Dict[str, StreamConfiguration] = {}
        self.active_tasks: Dict[str, RenderTask] = {}
        self.frame_metrics: List[FrameMetrics] = []
        
        # External connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.postgres_pool: Optional[asyncpg.Pool] = None
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Set[WebSocket] = set()
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.total_frames_rendered = 0
        self.total_frames_dropped = 0
        self.average_render_time_ms = 0.0
    
    async def initialize(self):
        """Initialize the render router service"""
        try:
            # Initialize GPU manager
            await self.gpu_manager.initialize()
            
            # Redis connection via centralized helper
            try:
                from utils.redis_helper import get_redis_client
                self.redis_client = await get_redis_client(os.getenv('REDIS_URL', 'redis://localhost:6379'))
            except Exception as e:
                logger.warning(f"Redis helper failed in render-router, using in-memory stub: {e}")
                class _InMemoryRedisStub:
                    def __init__(self):
                        self._store = {}
                    async def hset(self, key, mapping=None, **kwargs):
                        self._store[key] = mapping or kwargs
                    async def delete(self, key):
                        self._store.pop(key, None)
                    async def hgetall(self, key):
                        return self._store.get(key, {})
                self.redis_client = _InMemoryRedisStub()
            
            # PostgreSQL connection
            self.postgres_pool = await asyncpg.create_pool(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', '5432')),
                user=os.getenv('POSTGRES_USER', 'omega'),
                password=os.getenv('POSTGRES_PASSWORD', 'omega_secure_2025'),
                database=os.getenv('POSTGRES_DB', 'omega_sessions'),
                min_size=5,
                max_size=20
            )
            
            # Start background tasks
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.optimization_task = asyncio.create_task(self._optimization_loop())
            
            logger.info("Render router initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize render router", error=str(e))
            raise
    
    async def create_render_stream(self, stream_config: StreamConfiguration) -> Dict[str, Any]:
        """Create a new render stream"""
        try:
            # Validate configuration
            if not stream_config.displays:
                raise HTTPException(status_code=400, detail="No displays configured")
            
            # Select optimal encoding profile
            encoding_profile = self.encoding_optimizer.select_optimal_profile(stream_config)
            
            # Store stream configuration
            self.active_streams[stream_config.stream_id] = stream_config
            
            # Update metrics
            active_streams.inc()
            
            # Notify via WebSocket
            await self._broadcast_stream_update("stream_created", stream_config.stream_id)
            
            logger.info(
                "Render stream created",
                stream_id=stream_config.stream_id,
                resolution=f"{stream_config.target_resolution[0]}x{stream_config.target_resolution[1]}",
                fps=stream_config.target_fps,
                codec=encoding_profile["codec"].value
            )
            
            return {
                "stream_id": stream_config.stream_id,
                "encoding_profile": encoding_profile,
                "status": "created"
            }
            
        except Exception as e:
            logger.error("Failed to create render stream", error=str(e))
            raise
    
    async def submit_render_task(self, render_task: RenderTask) -> Dict[str, Any]:
        """Submit a rendering task"""
        try:
            start_time = time.time()
            
            # Select optimal GPU
            selected_gpu = await self.gpu_manager.select_optimal_gpu(render_task)
            if not selected_gpu:
                frame_drops_total.labels(reason="no_gpu_available").inc()
                raise HTTPException(status_code=503, detail="No suitable GPU available")
            
            # Add task to GPU queue
            await self.gpu_manager.gpu_queues[selected_gpu].put(render_task)
            self.active_tasks[render_task.task_id] = render_task
            
            # Simulate rendering process
            render_result = await self._execute_render_task(render_task, selected_gpu)
            
            # Update metrics
            render_duration = (time.time() - start_time) * 1000
            frame_render_duration.observe(render_duration)
            render_requests_total.labels(gpu_id=selected_gpu, encoding="h264").inc()
            
            # Create frame metrics
            frame_metrics = FrameMetrics(
                frame_number=render_task.frame_number,
                stream_id=render_task.stream_id,
                timestamp=datetime.utcnow(),
                render_start_time=start_time,
                render_end_time=time.time(),
                encode_start_time=time.time(),
                encode_end_time=time.time() + 0.005,  # 5ms encoding
                transmission_start_time=time.time() + 0.005,
                frame_size_bytes=render_result["frame_size_bytes"],
                quality_score=render_result["quality_score"],
                motion_intensity=0.5,
                complexity_score=0.6,
                gpu_utilization_during_render=self.gpu_manager.gpu_capabilities[selected_gpu].current_utilization_percent,
                memory_usage_mb=render_result["memory_used_mb"],
                dropped=False,
                drop_reason=None
            )
            
            self.frame_metrics.append(frame_metrics)
            self.total_frames_rendered += 1
            
            # Clean up old metrics
            if len(self.frame_metrics) > 1000:
                self.frame_metrics = self.frame_metrics[-500:]
            
            # Remove from active tasks
            del self.active_tasks[render_task.task_id]
            
            return {
                "task_id": render_task.task_id,
                "gpu_used": selected_gpu,
                "render_time_ms": render_duration,
                "frame_size_bytes": render_result["frame_size_bytes"],
                "quality_score": render_result["quality_score"]
            }
            
        except Exception as e:
            self.total_frames_dropped += 1
            frame_drops_total.labels(reason="render_error").inc()
            logger.error("Render task failed", task_id=render_task.task_id, error=str(e))
            raise
    
    async def _execute_render_task(self, task: RenderTask, gpu_id: str) -> Dict[str, Any]:
        """Execute the actual rendering (simulated)"""
        # Simulate rendering process
        viewport_area = task.viewport[2] * task.viewport[3]
        complexity_factor = viewport_area / (1920 * 1080)
        
        # Simulate render time based on complexity
        base_render_time = 8.0  # ms
        render_time = base_render_time * complexity_factor
        
        await asyncio.sleep(render_time / 1000.0)  # Convert to seconds
        
        # Simulate frame data
        frame_size = int(viewport_area * 0.1)  # Rough compression estimate
        quality_score = 0.85  # Good quality
        memory_used = int(viewport_area * 4 / (1024 * 1024))  # MB
        
        return {
            "frame_size_bytes": frame_size,
            "quality_score": quality_score,
            "memory_used_mb": memory_used
        }
    
    async def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """Get status of a render stream"""
        if stream_id not in self.active_streams:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        stream_config = self.active_streams[stream_id]
        
        # Get recent frame metrics for this stream
        recent_metrics = [
            m for m in self.frame_metrics[-100:]
            if m.stream_id == stream_id
        ]
        
        if recent_metrics:
            avg_render_time = sum(
                (m.render_end_time - m.render_start_time) * 1000
                for m in recent_metrics
            ) / len(recent_metrics)
            
            avg_quality = sum(m.quality_score for m in recent_metrics) / len(recent_metrics)
            
            frame_drop_rate = sum(1 for m in recent_metrics if m.dropped) / len(recent_metrics)
        else:
            avg_render_time = 0.0
            avg_quality = 0.0
            frame_drop_rate = 0.0
        
        return {
            "stream_id": stream_id,
            "status": "active",
            "target_resolution": stream_config.target_resolution,
            "target_fps": stream_config.target_fps,
            "encoding_codec": stream_config.encoding_codec.value,
            "current_metrics": {
                "average_render_time_ms": avg_render_time,
                "average_quality_score": avg_quality,
                "frame_drop_rate": frame_drop_rate,
                "frames_rendered": len(recent_metrics)
            }
        }
    
    async def optimize_stream(self, stream_id: str) -> Dict[str, Any]:
        """Optimize encoding parameters for a stream"""
        if stream_id not in self.active_streams:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        stream_config = self.active_streams[stream_id]
        
        # Analyze recent performance
        recent_metrics = [
            m for m in self.frame_metrics[-50:]
            if m.stream_id == stream_id
        ]
        
        if not recent_metrics:
            return {"message": "Insufficient data for optimization"}
        
        optimizations = []
        
        # Check render time
        avg_render_time = sum(
            (m.render_end_time - m.render_start_time) * 1000
            for m in recent_metrics
        ) / len(recent_metrics)
        
        if avg_render_time > 16.67:  # > 60fps budget
            optimizations.append("Reduce rendering quality to meet frame time budget")
        
        # Check frame drops
        frame_drops = sum(1 for m in recent_metrics if m.dropped)
        if frame_drops > len(recent_metrics) * 0.05:  # > 5% drop rate
            optimizations.append("Increase GPU allocation or reduce target FPS")
        
        # Check quality
        avg_quality = sum(m.quality_score for m in recent_metrics) / len(recent_metrics)
        if avg_quality < 0.7:
            optimizations.append("Increase bitrate or improve encoding settings")
        
        return {
            "stream_id": stream_id,
            "current_performance": {
                "average_render_time_ms": avg_render_time,
                "frame_drop_rate": frame_drops / len(recent_metrics),
                "average_quality": avg_quality
            },
            "optimizations": optimizations
        }
    
    async def _monitoring_loop(self):
        """Continuous monitoring of render performance"""
        while True:
            try:
                # Update GPU metrics
                await self.gpu_manager.update_gpu_metrics()
                
                # Update global metrics
                if self.total_frames_rendered > 0:
                    overall_drop_rate = self.total_frames_dropped / self.total_frames_rendered
                    
                    # Log performance summary
                    if self.total_frames_rendered % 1000 == 0:
                        logger.info(
                            "Render performance summary",
                            total_frames=self.total_frames_rendered,
                            total_drops=self.total_frames_dropped,
                            drop_rate=overall_drop_rate,
                            active_streams=len(self.active_streams)
                        )
                
                await asyncio.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(10.0)
    
    async def _optimization_loop(self):
        """Continuous optimization of rendering parameters"""
        while True:
            try:
                # Run optimization for all active streams
                for stream_id in list(self.active_streams.keys()):
                    try:
                        await self.optimize_stream(stream_id)
                    except Exception as e:
                        logger.error("Stream optimization failed", stream_id=stream_id, error=str(e))
                
                await asyncio.sleep(30.0)  # Optimize every 30 seconds
                
            except Exception as e:
                logger.error("Error in optimization loop", error=str(e))
                await asyncio.sleep(60.0)
    
    async def _broadcast_stream_update(self, event_type: str, stream_id: str):
        """Broadcast stream updates to WebSocket clients"""
        if self.websocket_connections:
            message = {
                "type": event_type,
                "stream_id": stream_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            disconnected = set()
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_json(message)
                except:
                    disconnected.add(websocket)
            
            self.websocket_connections -= disconnected
    
    async def add_websocket_connection(self, websocket: WebSocket):
        """Add WebSocket connection"""
        self.websocket_connections.add(websocket)
    
    async def remove_websocket_connection(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.websocket_connections.discard(websocket)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        if self.optimization_task:
            self.optimization_task.cancel()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.postgres_pool:
            await self.postgres_pool.close()
        
        logger.info("Render router cleanup completed")

# FastAPI Application
app = FastAPI(
    title="Omega Render Router",
    description="Advanced graphics rendering routing and optimization service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global render router instance
render_router = RenderRouter()

@app.on_event("startup")
async def startup_event():
    """Initialize the render router"""
    await render_router.initialize()
    
    # Start Prometheus metrics server
    start_http_server(8003)
    logger.info("Render router started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await render_router.cleanup()

# API Endpoints
@app.post("/streams", response_model=Dict[str, Any])
async def create_render_stream(request: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new render stream"""
    try:
        # Parse display configurations
        displays = []
        for display_data in request.get('displays', []):
            display = DisplayConfiguration(
                display_id=display_data['display_id'],
                resolution_width=display_data['resolution_width'],
                resolution_height=display_data['resolution_height'],
                refresh_rate=display_data.get('refresh_rate', 60.0),
                color_depth=display_data.get('color_depth', 8),
                color_space=ColorSpace(display_data.get('color_space', 'rec709')),
                hdr_enabled=display_data.get('hdr_enabled', False),
                variable_refresh_rate=display_data.get('variable_refresh_rate', False),
                adaptive_sync=display_data.get('adaptive_sync', False),
                primary_display=display_data.get('primary_display', True)
            )
            displays.append(display)
        
        # Create stream configuration
        stream_config = StreamConfiguration(
            stream_id=request['stream_id'],
            session_id=request['session_id'],
            client_id=request['client_id'],
            target_resolution=(request['target_resolution']['width'], request['target_resolution']['height']),
            target_fps=request.get('target_fps', 60),
            encoding_codec=EncodingCodec(request.get('encoding_codec', 'h264')),
            encoding_preset=EncodingPreset(request.get('encoding_preset', 'fast')),
            bitrate_mbps=request.get('bitrate_mbps', 50.0),
            keyframe_interval=request.get('keyframe_interval', 4),
            quality_level=request.get('quality_level', 0.8),
            adaptive_quality=request.get('adaptive_quality', True),
            low_latency_mode=request.get('low_latency_mode', False),
            client_capabilities=request.get('client_capabilities', {}),
            network_bandwidth_mbps=request.get('network_bandwidth_mbps', 100.0),
            network_latency_ms=request.get('network_latency_ms', 20.0),
            displays=displays,
            hdr_tone_mapping=request.get('hdr_tone_mapping', False),
            motion_vectors=request.get('motion_vectors', False),
            frame_pacing=request.get('frame_pacing', True),
            vsync_enabled=request.get('vsync_enabled', True)
        )
        
        result = await render_router.create_render_stream(stream_config)
        return result
        
    except Exception as e:
        logger.error("Failed to create render stream", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/render", response_model=Dict[str, Any])
async def submit_render_task(request: Dict[str, Any]) -> Dict[str, Any]:
    """Submit a rendering task"""
    try:
        render_task = RenderTask(
            task_id=request['task_id'],
            stream_id=request['stream_id'],
            frame_number=request['frame_number'],
            timestamp=datetime.utcnow(),
            viewport=(
                request['viewport']['x'],
                request['viewport']['y'],
                request['viewport']['width'],
                request['viewport']['height']
            ),
            rendering_api=RenderingAPI(request.get('rendering_api', 'directx12')),
            render_targets=request.get('render_targets', ['main']),
            max_render_time_ms=request.get('max_render_time_ms', 16.67),
            priority_level=request.get('priority_level', 1),
            application_id=request.get('application_id', 'unknown'),
            window_handle=request.get('window_handle'),
            gpu_context=request.get('gpu_context')
        )
        
        result = await render_router.submit_render_task(render_task)
        return result
        
    except Exception as e:
        logger.error("Failed to submit render task", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/streams/{stream_id}/status")
async def get_stream_status(stream_id: str) -> Dict[str, Any]:
    """Get render stream status"""
    return await render_router.get_stream_status(stream_id)

@app.post("/streams/{stream_id}/optimize")
async def optimize_stream(stream_id: str) -> Dict[str, Any]:
    """Optimize render stream"""
    return await render_router.optimize_stream(stream_id)

@app.get("/gpus")
async def list_gpus() -> Dict[str, Any]:
    """List available GPUs and their capabilities"""
    return {
        "gpus": [
            asdict(capabilities) 
            for capabilities in render_router.gpu_manager.gpu_capabilities.values()
        ]
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time render updates"""
    await websocket.accept()
    await render_router.add_websocket_connection(websocket)
    
    try:
        while True:
            await websocket.receive_text()
    except:
        pass
    finally:
        await render_router.remove_websocket_connection(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_streams": len(render_router.active_streams),
        "active_tasks": len(render_router.active_tasks),
        "total_frames_rendered": render_router.total_frames_rendered,
        "total_frames_dropped": render_router.total_frames_dropped,
        "available_gpus": len(render_router.gpu_manager.gpu_capabilities),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Get detailed performance metrics"""
    drop_rate = 0.0
    if render_router.total_frames_rendered > 0:
        drop_rate = render_router.total_frames_dropped / render_router.total_frames_rendered
    
    return {
        "total_frames_rendered": render_router.total_frames_rendered,
        "total_frames_dropped": render_router.total_frames_dropped,
        "frame_drop_rate": drop_rate,
        "active_streams": len(render_router.active_streams),
        "active_tasks": len(render_router.active_tasks),
        "gpu_utilization": {
            gpu_id: capabilities.current_utilization_percent
            for gpu_id, capabilities in render_router.gpu_manager.gpu_capabilities.items()
        },
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8005,
        reload=False,
        access_log=True
    )
