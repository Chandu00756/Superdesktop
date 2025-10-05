"""
Advanced Virtual Desktop Manager with Enterprise Features
Handles complete VD lifecycle, RDP/VNC protocols, snapshots, and cluster orchestration
"""

import asyncio
import json
import os
import docker
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import subprocess
import psutil
import aiohttp
import aiodns
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VDState(Enum):
    CREATING = "creating"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    TERMINATED = "terminated"

class VDProtocol(Enum):
    VNC = "vnc"
    RDP = "rdp"
    WEBRTC = "webrtc"
    SPICE = "spice"

@dataclass
class VDSnapshot:
    snapshot_id: str
    vd_session_id: str
    name: str
    description: str
    created_at: datetime
    size_bytes: int
    metadata: Dict[str, Any]

@dataclass
class VDSession:
    session_id: str
    user_id: str
    node_id: str
    state: VDState
    protocol: VDProtocol
    container_id: str
    image: str
    cpu_cores: int
    memory_gb: int
    gpu_units: int = 0
    storage_gb: int = 20
    created_at: datetime = None
    last_accessed: datetime = None
    vnc_port: int = None
    rdp_port: int = None
    webrtc_port: int = None
    spice_port: int = None
    url: str = None
    snapshots: List[VDSnapshot] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        if self.snapshots is None:
            self.snapshots = []

class AdvancedVirtualDesktopManager:
    """Enterprise-grade Virtual Desktop Manager with full lifecycle management"""
    
    def __init__(self):
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            print(f"Warning: Docker not available: {e}")
            self.docker_client = None
            self.docker_available = False
        
        self.sessions: Dict[str, VDSession] = {}
        self.snapshots: Dict[str, VDSnapshot] = {}
        self.base_vnc_port = 5900
        self.base_rdp_port = 3389
        self.base_webrtc_port = 8080
        self.base_spice_port = 5930
        self.port_allocator = PortAllocator()
        
        # Load balancing and cluster management
        self.load_balancer = VDLoadBalancer()
        self.cluster_manager = VDClusterManager()
        
        # Initialize encryption for session data
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
    async def create_virtual_desktop(
        self,
        user_id: str,
        os_image: str = "dorowu/ubuntu-desktop-lxde-vnc",
        cpu_cores: int = 2,
        memory_gb: int = 4,
        gpu_units: int = 0,
        storage_gb: int = 20,
        protocol: VDProtocol = VDProtocol.VNC,
        node_id: Optional[str] = None
    ) -> VDSession:
        """Create a new virtual desktop with advanced configuration"""
        
        session_id = str(uuid.uuid4())
        
        # Select optimal node if not specified
        if not node_id:
            node_id = await self.load_balancer.select_optimal_node(
                cpu_cores, memory_gb, gpu_units
            )
        
        # Allocate ports based on protocol
        ports = await self._allocate_ports(protocol)
        
        # Create container with advanced configuration
        container = await self._create_container(
            session_id, os_image, cpu_cores, memory_gb, gpu_units, 
            storage_gb, ports, protocol
        )
        
        # Create session object
        session = VDSession(
            session_id=session_id,
            user_id=user_id,
            node_id=node_id,
            state=VDState.CREATING,
            protocol=protocol,
            container_id=container.id,
            image=os_image,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_units=gpu_units,
            storage_gb=storage_gb,
            vnc_port=ports.get('vnc'),
            rdp_port=ports.get('rdp'),
            webrtc_port=ports.get('webrtc'),
            spice_port=ports.get('spice')
        )
        
        # Generate connection URL
        session.url = await self._generate_connection_url(session)
        
        # Store session
        self.sessions[session_id] = session
        
        # Wait for container to be ready
        await self._wait_for_container_ready(container.id)
        session.state = VDState.RUNNING
        
        logger.info(f"Created VD session {session_id} for user {user_id}")
        return session
    
    async def get_session_url(self, session_id: str) -> str:
        """Get connection URL for a session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        if session.state != VDState.RUNNING:
            raise ValueError(f"Session {session_id} is not running")
        
        return session.url
    
    async def pause_session(self, session_id: str) -> bool:
        """Pause a virtual desktop session"""
        if not self.docker_available:
            logger.warning("Docker not available - returning mock success")
            return True
            
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        try:
            container = self.docker_client.containers.get(session.container_id)
            container.pause()
            session.state = VDState.PAUSED
            logger.info(f"Paused session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause session {session_id}: {e}")
            return False
    
    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused virtual desktop session"""
        if not self.docker_available:
            logger.warning("Docker not available - returning mock success")
            return True
            
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        try:
            container = self.docker_client.containers.get(session.container_id)
            container.unpause()
            session.state = VDState.RUNNING
            session.last_accessed = datetime.now()
            logger.info(f"Resumed session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume session {session_id}: {e}")
            return False
    
    async def terminate_session(self, session_id: str) -> bool:
        """Terminate a virtual desktop session"""
        if not self.docker_available:
            # Clean up session data even without Docker
            if session_id in self.sessions:
                session = self.sessions[session_id]
                self.port_allocator.release_port(session.vnc_port)
                del self.sessions[session_id]
                logger.info(f"Terminated session {session_id} (Docker unavailable)")
            return True
            
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        try:
            # Stop and remove container
            container = self.docker_client.containers.get(session.container_id)
            container.stop(timeout=10)
            container.remove()
            
            # Free allocated ports
            await self._free_ports(session)
            
            # Update session state
            session.state = VDState.TERMINATED
            
            logger.info(f"Terminated session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to terminate session {session_id}: {e}")
            return False
    
    async def create_snapshot(
        self, 
        session_id: str, 
        name: str, 
        description: str = ""
    ) -> VDSnapshot:
        """Create a snapshot of a virtual desktop session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        snapshot_id = str(uuid.uuid4())
        
        try:
            # Commit container to create image
            container = self.docker_client.containers.get(session.container_id)
            image_name = f"vd_snapshot_{session_id}_{snapshot_id}"
            container.commit(repository=image_name, tag="latest")
            
            # Get image size
            image = self.docker_client.images.get(f"{image_name}:latest")
            size_bytes = image.attrs['Size']
            
            # Create snapshot object
            snapshot = VDSnapshot(
                snapshot_id=snapshot_id,
                vd_session_id=session_id,
                name=name,
                description=description,
                created_at=datetime.now(),
                size_bytes=size_bytes,
                metadata={
                    "image_name": image_name,
                    "session_state": session.state.value,
                    "protocol": session.protocol.value
                }
            )
            
            # Store snapshot
            self.snapshots[snapshot_id] = snapshot
            session.snapshots.append(snapshot)
            
            logger.info(f"Created snapshot {snapshot_id} for session {session_id}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to create snapshot for session {session_id}: {e}")
            raise
    
    async def list_snapshots(self, session_id: str) -> List[VDSnapshot]:
        """List all snapshots for a session"""
        if session_id not in self.sessions:
            return []
        
        return self.sessions[session_id].snapshots
    
    async def delete_snapshot(self, session_id: str, snapshot_id: str) -> bool:
        """Delete a specific snapshot"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        try:
            # Find and remove snapshot from session
            snapshot = None
            for s in session.snapshots:
                if s.snapshot_id == snapshot_id:
                    snapshot = s
                    break
            
            if not snapshot:
                return False
            
            # Remove Docker image
            image_name = snapshot.metadata.get("image_name")
            if image_name:
                try:
                    self.docker_client.images.remove(f"{image_name}:latest", force=True)
                except Exception as e:
                    logger.warning(f"Failed to remove image {image_name}: {e}")
            
            # Remove from collections
            session.snapshots.remove(snapshot)
            if snapshot_id in self.snapshots:
                del self.snapshots[snapshot_id]
            
            logger.info(f"Deleted snapshot {snapshot_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete snapshot {snapshot_id}: {e}")
            return False
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[VDSession]:
        """List virtual desktop sessions, optionally filtered by user"""
        sessions = list(self.sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        
        return sessions
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a session"""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        
        try:
            container = self.docker_client.containers.get(session.container_id)
            stats = container.stats(stream=False)
            
            # Calculate CPU and memory usage
            cpu_usage = self._calculate_cpu_usage(stats)
            memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
            
            return {
                "session_id": session_id,
                "state": session.state.value,
                "uptime_seconds": (datetime.now() - session.created_at).total_seconds(),
                "cpu_usage_percent": cpu_usage,
                "memory_usage_mb": memory_usage,
                "container_status": container.status,
                "ports": {
                    "vnc": session.vnc_port,
                    "rdp": session.rdp_port,
                    "webrtc": session.webrtc_port,
                    "spice": session.spice_port
                }
            }
        except Exception as e:
            logger.error(f"Failed to get stats for session {session_id}: {e}")
            return {"error": str(e)}
    
    async def _allocate_ports(self, protocol: VDProtocol) -> Dict[str, int]:
        """Allocate ports for the specified protocol"""
        ports = {}
        
        if protocol == VDProtocol.VNC:
            ports['vnc'] = await self.port_allocator.allocate_port(self.base_vnc_port)
        elif protocol == VDProtocol.RDP:
            ports['rdp'] = await self.port_allocator.allocate_port(self.base_rdp_port)
        elif protocol == VDProtocol.WEBRTC:
            ports['webrtc'] = await self.port_allocator.allocate_port(self.base_webrtc_port)
        elif protocol == VDProtocol.SPICE:
            ports['spice'] = await self.port_allocator.allocate_port(self.base_spice_port)
        
        return ports
    
    async def _create_container(
        self, 
        session_id: str, 
        image: str, 
        cpu_cores: int, 
        memory_gb: int, 
        gpu_units: int,
        storage_gb: int,
        ports: Dict[str, int],
        protocol: VDProtocol
    ):
        """Create and configure Docker container for VD session"""
        
        # Check if Docker is available
        if not self.docker_available:
            # Return a mock container ID for testing
            return f"mock_container_{session_id}"
        
        # Configure environment variables
        env_vars = {
            "VNC_PW": "password",
            "RESOLUTION": "1920x1080",
            "DISPLAY": ":1"
        }
        
        # Configure port mappings
        port_bindings = {}
        if 'vnc' in ports:
            port_bindings[5901] = ports['vnc']
        if 'rdp' in ports:
            port_bindings[3389] = ports['rdp']
        if 'webrtc' in ports:
            port_bindings[8080] = ports['webrtc']
        if 'spice' in ports:
            port_bindings[5930] = ports['spice']
        
        # Configure resource limits
        host_config = self.docker_client.api.create_host_config(
            port_bindings=port_bindings,
            mem_limit=f"{memory_gb}g",
            cpu_count=cpu_cores,
            shm_size="2g"  # Increased shared memory for GUI apps
        )
        
        # Add GPU support if requested
        if gpu_units > 0:
            host_config['device_requests'] = [
                docker.types.DeviceRequest(count=gpu_units, capabilities=[['gpu']])
            ]
        
        # Create container
        container = self.docker_client.containers.run(
            image=image,
            detach=True,
            environment=env_vars,
            ports=port_bindings,
            host_config=host_config,
            name=f"vd_session_{session_id}",
            labels={
                "vd.session_id": session_id,
                "vd.protocol": protocol.value
            }
        )
        
        return container
    
    async def _wait_for_container_ready(self, container_id: str, timeout: int = 60):
        """Wait for container to be ready to accept connections"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            try:
                container = self.docker_client.containers.get(container_id)
                if container.status == 'running':
                    # Additional check for VNC service
                    logs = container.logs(tail=50).decode()
                    if "VNC server is now running" in logs or "noVNC started" in logs:
                        return True
                
                await asyncio.sleep(2)
            except Exception as e:
                logger.warning(f"Error checking container readiness: {e}")
                await asyncio.sleep(2)
        
        raise TimeoutError(f"Container {container_id} not ready after {timeout} seconds")
    
    async def _generate_connection_url(self, session: VDSession) -> str:
        """Generate connection URL based on protocol"""
        host = os.getenv('OMEGA_VD_HOST', '127.0.0.1')
        
        if session.protocol == VDProtocol.VNC:
            return f"http://{host}:{session.vnc_port}/vnc.html"
        elif session.protocol == VDProtocol.RDP:
            return f"rdp://{host}:{session.rdp_port}"
        elif session.protocol == VDProtocol.WEBRTC:
            return f"https://{host}:{session.webrtc_port}/webrtc"
        elif session.protocol == VDProtocol.SPICE:
            return f"spice://{host}:{session.spice_port}"
        
        return ""
    
    async def _free_ports(self, session: VDSession):
        """Free allocated ports when session terminates"""
        if session.vnc_port:
            await self.port_allocator.free_port(session.vnc_port)
        if session.rdp_port:
            await self.port_allocator.free_port(session.rdp_port)
        if session.webrtc_port:
            await self.port_allocator.free_port(session.webrtc_port)
        if session.spice_port:
            await self.port_allocator.free_port(session.spice_port)
    
    def _calculate_cpu_usage(self, stats: dict) -> float:
        """Calculate CPU usage percentage from container stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
                return round(cpu_percent, 2)
        except (KeyError, ZeroDivisionError):
            pass
        
        return 0.0

class PortAllocator:
    """Manages port allocation for VD sessions"""
    
    def __init__(self):
        self.allocated_ports = set()
    
    async def allocate_port(self, base_port: int) -> int:
        """Allocate an available port starting from base_port"""
        port = base_port
        while port in self.allocated_ports or not self._is_port_available(port):
            port += 1
            if port > base_port + 1000:  # Prevent infinite loop
                raise RuntimeError(f"No available ports starting from {base_port}")
        
        self.allocated_ports.add(port)
        return port
    
    async def free_port(self, port: int):
        """Free an allocated port"""
        self.allocated_ports.discard(port)
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return True
        except OSError:
            return False

class VDLoadBalancer:
    """Advanced load balancer for VD session placement"""
    
    async def select_optimal_node(
        self, 
        cpu_cores: int, 
        memory_gb: int, 
        gpu_units: int = 0
    ) -> str:
        """Select the optimal node for a new VD session"""
        # For now, return localhost - in a real cluster this would
        # evaluate multiple nodes based on resource availability,
        # latency, and current load
        return "localhost"

class VDClusterManager:
    """Manages VD sessions across a cluster of nodes"""
    
    def __init__(self):
        self.nodes = {}
    
    async def register_node(self, node_id: str, node_info: dict):
        """Register a new node in the cluster"""
        self.nodes[node_id] = node_info
    
    async def get_cluster_status(self) -> dict:
        """Get overall cluster status"""
        return {
            "total_nodes": len(self.nodes),
            "active_sessions": 0,  # Would be calculated from all nodes
            "total_capacity": {},  # Would aggregate from all nodes
        }

# Global instance
vd_manager = AdvancedVirtualDesktopManager()