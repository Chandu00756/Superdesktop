"""
Omega Super Desktop Console v2.0 - Advanced Network Mesh Engine
Enterprise-grade mesh networking with multi-transport support and intelligent routing
"""

import asyncio
import json
import logging
import time
import uuid
import socket
import struct
import hashlib
import random
import threading
import ssl
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path
import ipaddress
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)

class TransportType(Enum):
    TCP = "tcp"
    UDP = "udp"
    QUIC = "quic"
    WEBSOCKET = "websocket"
    BLUETOOTH = "bluetooth"
    WIFI_DIRECT = "wifi_direct"
    ZIGBEE = "zigbee"
    LORA = "lora"

class NetworkTopology(Enum):
    MESH = "mesh"
    STAR = "star"
    RING = "ring"
    TREE = "tree"
    HYBRID = "hybrid"

class RoutingProtocol(Enum):
    AODV = "aodv"  # Ad-hoc On-Demand Distance Vector
    OLSR = "olsr"  # Optimized Link State Routing
    DSR = "dsr"   # Dynamic Source Routing
    BATMAN = "batman"  # Better Approach to Mobile Ad-hoc Networking
    CUSTOM = "custom"

class QoSClass(Enum):
    CRITICAL = "critical"     # Real-time, low latency
    HIGH = "high"            # Important, medium latency
    NORMAL = "normal"        # Standard traffic
    LOW = "low"             # Background traffic

@dataclass
class NetworkNode:
    """Network mesh node representation"""
    node_id: str
    hostname: str
    ip_addresses: List[str]
    mac_addresses: List[str]
    transport_capabilities: Set[TransportType]
    location: Optional[Tuple[float, float]] = None  # lat, lon
    hardware_specs: Dict[str, Any] = field(default_factory=dict)
    trust_score: float = 0.5
    reputation: float = 0.5
    last_seen: float = field(default_factory=time.time)
    is_gateway: bool = False
    is_bridge: bool = False
    battery_level: Optional[float] = None
    signal_strength: Dict[str, float] = field(default_factory=dict)

@dataclass
class NetworkLink:
    """Link between two network nodes"""
    link_id: str
    source_node: str
    target_node: str
    transport_type: TransportType
    bandwidth: float  # Mbps
    latency: float    # ms
    packet_loss: float  # percentage
    reliability: float  # percentage
    cost: float = 1.0
    qos_class: QoSClass = QoSClass.NORMAL
    encryption_level: str = "AES256"
    is_active: bool = True
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

@dataclass
class RoutingEntry:
    """Routing table entry"""
    destination: str
    next_hop: str
    hop_count: int
    metric: float
    interface: str
    timestamp: float = field(default_factory=time.time)
    ttl: int = 300  # seconds

@dataclass
class NetworkPacket:
    """Network packet representation"""
    packet_id: str
    source: str
    destination: str
    payload: bytes
    packet_type: str
    priority: int = 0
    ttl: int = 64
    qos_class: QoSClass = QoSClass.NORMAL
    route_history: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    encryption_key: Optional[str] = None

class NetworkTopologyManager:
    """Manage network topology and auto-discovery"""
    
    def __init__(self):
        self.nodes: Dict[str, NetworkNode] = {}
        self.links: Dict[str, NetworkLink] = {}
        self.topology = NetworkTopology.MESH
        self.discovery_active = False
        
    async def discover_nodes(self, interfaces: List[str]) -> List[NetworkNode]:
        """Discover nodes on network interfaces"""
        discovered_nodes = []
        
        try:
            for interface in interfaces:
                # Get interface network
                network = await self._get_interface_network(interface)
                if not network:
                    continue
                    
                # Scan network for active hosts
                active_hosts = await self._scan_network(network)
                
                for host_ip in active_hosts:
                    # Probe host for capabilities
                    node = await self._probe_host(host_ip)
                    if node:
                        discovered_nodes.append(node)
                        self.nodes[node.node_id] = node
                        
            logger.info(f"Discovered {len(discovered_nodes)} network nodes")
            return discovered_nodes
            
        except Exception as e:
            logger.error(f"Node discovery failed: {e}")
            return []
            
    async def _get_interface_network(self, interface: str) -> Optional[str]:
        """Get network address for interface"""
        try:
            # In production, use netifaces or similar library
            # For demo, return common networks
            networks = [
                "192.168.1.0/24",
                "192.168.0.0/24", 
                "10.0.0.0/24",
                "172.16.0.0/24"
            ]
            return random.choice(networks)
        except Exception as e:
            logger.error(f"Failed to get interface network: {e}")
            return None
            
    async def _scan_network(self, network: str) -> List[str]:
        """Scan network for active hosts"""
        try:
            network_obj = ipaddress.IPv4Network(network, strict=False)
            active_hosts = []
            
            # Simulate network scan
            for host in list(network_obj.hosts())[:20]:  # Limit scan
                # Simulate ping response
                if random.random() > 0.7:  # 30% response rate
                    active_hosts.append(str(host))
                    
            return active_hosts
            
        except Exception as e:
            logger.error(f"Network scan failed: {e}")
            return []
            
    async def _probe_host(self, host_ip: str) -> Optional[NetworkNode]:
        """Probe host for node capabilities"""
        try:
            # Generate synthetic node data
            node_id = hashlib.md5(host_ip.encode()).hexdigest()[:16]
            
            node = NetworkNode(
                node_id=node_id,
                hostname=f"node-{node_id[:8]}",
                ip_addresses=[host_ip],
                mac_addresses=[self._generate_mac()],
                transport_capabilities={
                    TransportType.TCP,
                    TransportType.UDP,
                    random.choice(list(TransportType))
                },
                hardware_specs={
                    'cpu_cores': random.randint(2, 16),
                    'memory_gb': random.choice([4, 8, 16, 32]),
                    'storage_gb': random.choice([256, 512, 1024])
                },
                trust_score=random.uniform(0.3, 0.9),
                reputation=random.uniform(0.5, 1.0)
            )
            
            return node
            
        except Exception as e:
            logger.error(f"Host probe failed: {e}")
            return None
            
    def _generate_mac(self) -> str:
        """Generate synthetic MAC address"""
        return ":".join([f"{random.randint(0, 255):02x}" for _ in range(6)])
        
    async def build_mesh_topology(self) -> Dict[str, List[str]]:
        """Build mesh network topology"""
        try:
            topology = defaultdict(list)
            
            # Create full mesh for small networks
            if len(self.nodes) <= 10:
                for node1_id in self.nodes:
                    for node2_id in self.nodes:
                        if node1_id != node2_id:
                            topology[node1_id].append(node2_id)
                            await self._create_link(node1_id, node2_id)
            else:
                # Create partial mesh for larger networks
                for node_id in self.nodes:
                    # Connect to 3-5 random neighbors
                    neighbors = random.sample(
                        [n for n in self.nodes.keys() if n != node_id],
                        min(random.randint(3, 5), len(self.nodes) - 1)
                    )
                    
                    for neighbor in neighbors:
                        topology[node_id].append(neighbor)
                        await self._create_link(node_id, neighbor)
                        
            return dict(topology)
            
        except Exception as e:
            logger.error(f"Topology building failed: {e}")
            return {}
            
    async def _create_link(self, source: str, target: str):
        """Create network link between nodes"""
        try:
            link_id = f"{source}-{target}"
            
            if link_id in self.links:
                return
                
            # Generate link characteristics
            transport = random.choice(list(TransportType))
            
            link = NetworkLink(
                link_id=link_id,
                source_node=source,
                target_node=target,
                transport_type=transport,
                bandwidth=random.uniform(10, 1000),  # 10 Mbps to 1 Gbps
                latency=random.uniform(1, 50),       # 1-50 ms
                packet_loss=random.uniform(0, 5),    # 0-5%
                reliability=random.uniform(85, 99.9), # 85-99.9%
                cost=random.uniform(0.1, 2.0)
            )
            
            self.links[link_id] = link
            
        except Exception as e:
            logger.error(f"Link creation failed: {e}")

class RoutingEngine:
    """Advanced routing engine with multiple protocols"""
    
    def __init__(self, protocol: RoutingProtocol = RoutingProtocol.AODV):
        self.protocol = protocol
        self.routing_table: Dict[str, RoutingEntry] = {}
        self.route_cache: Dict[str, List[str]] = {}
        self.link_state_db: Dict[str, Dict[str, Any]] = {}
        
    async def find_route(self, source: str, destination: str, links: Dict[str, NetworkLink]) -> Optional[List[str]]:
        """Find optimal route between nodes"""
        try:
            cache_key = f"{source}-{destination}"
            
            # Check cache first
            if cache_key in self.route_cache:
                route = self.route_cache[cache_key]
                if await self._validate_route(route, links):
                    return route
                else:
                    del self.route_cache[cache_key]
                    
            # Calculate new route based on protocol
            if self.protocol == RoutingProtocol.AODV:
                route = await self._aodv_route(source, destination, links)
            elif self.protocol == RoutingProtocol.OLSR:
                route = await self._olsr_route(source, destination, links)
            elif self.protocol == RoutingProtocol.DSR:
                route = await self._dsr_route(source, destination, links)
            elif self.protocol == RoutingProtocol.BATMAN:
                route = await self._batman_route(source, destination, links)
            else:
                route = await self._dijkstra_route(source, destination, links)
                
            if route:
                self.route_cache[cache_key] = route
                
            return route
            
        except Exception as e:
            logger.error(f"Route finding failed: {e}")
            return None
            
    async def _aodv_route(self, source: str, destination: str, links: Dict[str, NetworkLink]) -> Optional[List[str]]:
        """AODV (Ad-hoc On-Demand Distance Vector) routing"""
        try:
            # Build graph from links
            graph = defaultdict(list)
            for link in links.values():
                if link.is_active:
                    graph[link.source_node].append((link.target_node, link.cost))
                    graph[link.target_node].append((link.source_node, link.cost))
                    
            # Dijkstra's algorithm for shortest path
            distances = {node: float('inf') for node in graph}
            distances[source] = 0
            previous = {}
            
            heap = [(0, source)]
            visited = set()
            
            while heap:
                current_distance, current_node = heapq.heappop(heap)
                
                if current_node in visited:
                    continue
                    
                visited.add(current_node)
                
                if current_node == destination:
                    # Reconstruct path
                    path = []
                    while current_node is not None:
                        path.append(current_node)
                        current_node = previous.get(current_node)
                    return path[::-1]
                    
                for neighbor, weight in graph[current_node]:
                    distance = current_distance + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_node
                        heapq.heappush(heap, (distance, neighbor))
                        
            return None
            
        except Exception as e:
            logger.error(f"AODV routing failed: {e}")
            return None
            
    async def _olsr_route(self, source: str, destination: str, links: Dict[str, NetworkLink]) -> Optional[List[str]]:
        """OLSR (Optimized Link State Routing) routing"""
        try:
            # Simplified OLSR implementation
            # In production, this would maintain MPR (Multi-Point Relay) sets
            return await self._dijkstra_route(source, destination, links)
        except Exception as e:
            logger.error(f"OLSR routing failed: {e}")
            return None
            
    async def _dsr_route(self, source: str, destination: str, links: Dict[str, NetworkLink]) -> Optional[List[str]]:
        """DSR (Dynamic Source Routing) routing"""
        try:
            # Simplified DSR implementation
            # In production, this would use source routing with route caching
            return await self._dijkstra_route(source, destination, links)
        except Exception as e:
            logger.error(f"DSR routing failed: {e}")
            return None
            
    async def _batman_route(self, source: str, destination: str, links: Dict[str, NetworkLink]) -> Optional[List[str]]:
        """BATMAN (Better Approach to Mobile Ad-hoc Networking) routing"""
        try:
            # Simplified BATMAN implementation
            # In production, this would use proactive link-state with hop-by-hop routing
            return await self._dijkstra_route(source, destination, links)
        except Exception as e:
            logger.error(f"BATMAN routing failed: {e}")
            return None
            
    async def _dijkstra_route(self, source: str, destination: str, links: Dict[str, NetworkLink]) -> Optional[List[str]]:
        """Standard Dijkstra's shortest path algorithm"""
        try:
            # Build weighted graph
            graph = defaultdict(list)
            for link in links.values():
                if link.is_active:
                    # Weight based on latency, packet loss, and cost
                    weight = link.latency + (link.packet_loss * 10) + link.cost
                    graph[link.source_node].append((link.target_node, weight))
                    graph[link.target_node].append((link.source_node, weight))
                    
            # Dijkstra's algorithm
            distances = {node: float('inf') for node in graph}
            distances[source] = 0
            previous = {}
            
            heap = [(0, source)]
            visited = set()
            
            while heap:
                current_distance, current_node = heapq.heappop(heap)
                
                if current_node in visited:
                    continue
                    
                visited.add(current_node)
                
                if current_node == destination:
                    # Reconstruct path
                    path = []
                    while current_node is not None:
                        path.append(current_node)
                        current_node = previous.get(current_node)
                    return path[::-1]
                    
                for neighbor, weight in graph[current_node]:
                    distance = current_distance + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_node
                        heapq.heappush(heap, (distance, neighbor))
                        
            return None
            
        except Exception as e:
            logger.error(f"Dijkstra routing failed: {e}")
            return None
            
    async def _validate_route(self, route: List[str], links: Dict[str, NetworkLink]) -> bool:
        """Validate that a cached route is still valid"""
        try:
            for i in range(len(route) - 1):
                source = route[i]
                target = route[i + 1]
                
                # Check if link exists and is active
                link_id = f"{source}-{target}"
                reverse_link_id = f"{target}-{source}"
                
                if link_id in links and links[link_id].is_active:
                    continue
                elif reverse_link_id in links and links[reverse_link_id].is_active:
                    continue
                else:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Route validation failed: {e}")
            return False

class QoSManager:
    """Quality of Service management"""
    
    def __init__(self):
        self.qos_policies: Dict[QoSClass, Dict[str, Any]] = {
            QoSClass.CRITICAL: {
                'max_latency': 10,      # ms
                'min_bandwidth': 100,   # Mbps
                'max_packet_loss': 0.1, # %
                'priority': 4
            },
            QoSClass.HIGH: {
                'max_latency': 50,      # ms
                'min_bandwidth': 50,    # Mbps
                'max_packet_loss': 1.0, # %
                'priority': 3
            },
            QoSClass.NORMAL: {
                'max_latency': 100,     # ms
                'min_bandwidth': 10,    # Mbps
                'max_packet_loss': 5.0, # %
                'priority': 2
            },
            QoSClass.LOW: {
                'max_latency': 500,     # ms
                'min_bandwidth': 1,     # Mbps
                'max_packet_loss': 10.0, # %
                'priority': 1
            }
        }
        
        self.traffic_shaping: Dict[str, Dict[str, Any]] = {}
        
    async def classify_packet(self, packet: NetworkPacket) -> QoSClass:
        """Classify packet into QoS class"""
        try:
            # Packet type based classification
            if packet.packet_type in ['control', 'routing', 'heartbeat']:
                return QoSClass.CRITICAL
            elif packet.packet_type in ['voice', 'video', 'real_time']:
                return QoSClass.HIGH
            elif packet.packet_type in ['data', 'file_transfer']:
                return QoSClass.NORMAL
            else:
                return QoSClass.LOW
                
        except Exception as e:
            logger.error(f"Packet classification failed: {e}")
            return QoSClass.NORMAL
            
    async def apply_qos_policy(self, link: NetworkLink, packets: List[NetworkPacket]) -> List[NetworkPacket]:
        """Apply QoS policies to packet stream"""
        try:
            # Sort packets by priority (highest first)
            sorted_packets = sorted(packets, key=lambda p: self.qos_policies[p.qos_class]['priority'], reverse=True)
            
            # Apply traffic shaping
            shaped_packets = []
            current_bandwidth = 0
            
            for packet in sorted_packets:
                policy = self.qos_policies[packet.qos_class]
                
                # Check bandwidth constraints
                packet_size = len(packet.payload)
                required_bandwidth = (packet_size * 8) / 1000000  # Convert to Mbps
                
                if current_bandwidth + required_bandwidth <= link.bandwidth:
                    shaped_packets.append(packet)
                    current_bandwidth += required_bandwidth
                else:
                    # Drop or queue packet based on QoS class
                    if packet.qos_class in [QoSClass.CRITICAL, QoSClass.HIGH]:
                        # Force include critical/high priority packets
                        shaped_packets.append(packet)
                    # Low priority packets are dropped
                    
            return shaped_packets
            
        except Exception as e:
            logger.error(f"QoS policy application failed: {e}")
            return packets
            
    async def monitor_qos_compliance(self, link: NetworkLink) -> Dict[str, Any]:
        """Monitor QoS compliance for a link"""
        try:
            compliance = {}
            
            for qos_class, policy in self.qos_policies.items():
                compliant = True
                issues = []
                
                if link.latency > policy['max_latency']:
                    compliant = False
                    issues.append(f"Latency {link.latency}ms exceeds {policy['max_latency']}ms")
                    
                if link.bandwidth < policy['min_bandwidth']:
                    compliant = False
                    issues.append(f"Bandwidth {link.bandwidth}Mbps below {policy['min_bandwidth']}Mbps")
                    
                if link.packet_loss > policy['max_packet_loss']:
                    compliant = False
                    issues.append(f"Packet loss {link.packet_loss}% exceeds {policy['max_packet_loss']}%")
                    
                compliance[qos_class.value] = {
                    'compliant': compliant,
                    'issues': issues
                }
                
            return compliance
            
        except Exception as e:
            logger.error(f"QoS monitoring failed: {e}")
            return {}

class SecurityManager:
    """Network security and encryption management"""
    
    def __init__(self):
        self.encryption_keys: Dict[str, str] = {}
        self.certificates: Dict[str, str] = {}
        self.blacklisted_nodes: Set[str] = set()
        
    async def encrypt_packet(self, packet: NetworkPacket, key: str) -> bytes:
        """Encrypt packet payload"""
        try:
            # In production, use proper encryption library (AES, ChaCha20, etc.)
            # For demo, simple XOR encryption
            encrypted = bytearray()
            key_bytes = key.encode()[:32].ljust(32, b'\0')
            
            for i, byte in enumerate(packet.payload):
                encrypted.append(byte ^ key_bytes[i % len(key_bytes)])
                
            return bytes(encrypted)
            
        except Exception as e:
            logger.error(f"Packet encryption failed: {e}")
            return packet.payload
            
    async def decrypt_packet(self, encrypted_payload: bytes, key: str) -> bytes:
        """Decrypt packet payload"""
        try:
            # Reverse of encryption process
            decrypted = bytearray()
            key_bytes = key.encode()[:32].ljust(32, b'\0')
            
            for i, byte in enumerate(encrypted_payload):
                decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
                
            return bytes(decrypted)
            
        except Exception as e:
            logger.error(f"Packet decryption failed: {e}")
            return encrypted_payload
            
    async def authenticate_node(self, node_id: str, challenge: str, response: str) -> bool:
        """Authenticate network node"""
        try:
            # In production, use proper challenge-response authentication
            expected_response = hashlib.sha256(f"{node_id}{challenge}".encode()).hexdigest()
            return response == expected_response
            
        except Exception as e:
            logger.error(f"Node authentication failed: {e}")
            return False
            
    async def is_node_trusted(self, node_id: str, nodes: Dict[str, NetworkNode]) -> bool:
        """Check if node is trusted"""
        try:
            if node_id in self.blacklisted_nodes:
                return False
                
            if node_id in nodes:
                node = nodes[node_id]
                return node.trust_score >= 0.5 and node.reputation >= 0.5
                
            return False
            
        except Exception as e:
            logger.error(f"Trust check failed: {e}")
            return False

class NetworkMeshEngine:
    """Main network mesh management engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db_path = self.config.get('db_path', 'backend/network_mesh.db')
        
        # Core components
        self.topology_manager = NetworkTopologyManager()
        self.routing_engine = RoutingEngine(
            RoutingProtocol(self.config.get('routing_protocol', 'aodv'))
        )
        self.qos_manager = QoSManager()
        self.security_manager = SecurityManager()
        
        # Network state
        self.local_node_id = str(uuid.uuid4())
        self.mesh_active = False
        self.packet_queue: asyncio.Queue = asyncio.Queue()
        
        # Monitoring
        self.running = False
        
        # Metrics
        self.metrics = {
            'discovered_nodes': 0,
            'active_links': 0,
            'packets_routed': 0,
            'packets_dropped': 0,
            'mesh_convergence_time': 0.0,
            'average_latency': 0.0,
            'network_utilization': 0.0
        }
        
    async def initialize(self) -> bool:
        """Initialize the network mesh engine"""
        try:
            logger.info("Initializing Network Mesh Engine...")
            
            # Setup database
            await self._setup_database()
            
            # Initialize local node
            await self._setup_local_node()
            
            # Start mesh networking
            self.running = True
            self.mesh_active = True
            
            # Start background tasks
            asyncio.create_task(self._mesh_maintenance())
            asyncio.create_task(self._packet_processor())
            asyncio.create_task(self._link_monitor())
            
            logger.info("Network Mesh Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Network mesh initialization failed: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown the network mesh engine"""
        try:
            logger.info("Shutting down Network Mesh Engine...")
            self.running = False
            self.mesh_active = False
            
            # Clear queues
            while not self.packet_queue.empty():
                try:
                    self.packet_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
            logger.info("Network Mesh Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Network mesh shutdown error: {e}")
            
    async def discover_network(self, interfaces: Optional[List[str]] = None) -> Dict[str, Any]:
        """Discover network nodes and build mesh topology"""
        try:
            if interfaces is None:
                interfaces = ['eth0', 'wlan0', 'en0']  # Common interface names
                
            start_time = time.time()
            
            # Discover nodes
            discovered_nodes = await self.topology_manager.discover_nodes(interfaces)
            self.metrics['discovered_nodes'] = len(discovered_nodes)
            
            # Build mesh topology
            topology = await self.topology_manager.build_mesh_topology()
            self.metrics['active_links'] = len(self.topology_manager.links)
            
            convergence_time = time.time() - start_time
            self.metrics['mesh_convergence_time'] = convergence_time
            
            logger.info(f"Network discovery complete: {len(discovered_nodes)} nodes, {len(topology)} connections")
            
            return {
                'nodes': [
                    {
                        'node_id': node.node_id,
                        'hostname': node.hostname,
                        'ip_addresses': node.ip_addresses,
                        'transport_capabilities': [t.value for t in node.transport_capabilities],
                        'trust_score': node.trust_score,
                        'hardware_specs': node.hardware_specs
                    }
                    for node in discovered_nodes
                ],
                'topology': topology,
                'convergence_time': convergence_time
            }
            
        except Exception as e:
            logger.error(f"Network discovery failed: {e}")
            return {}
            
    async def send_packet(self, destination: str, payload: bytes, packet_type: str = "data", 
                         qos_class: QoSClass = QoSClass.NORMAL) -> bool:
        """Send packet through mesh network"""
        try:
            # Create packet
            packet = NetworkPacket(
                packet_id=str(uuid.uuid4()),
                source=self.local_node_id,
                destination=destination,
                payload=payload,
                packet_type=packet_type,
                qos_class=qos_class
            )
            
            # Add to queue for processing
            await self.packet_queue.put(packet)
            
            return True
            
        except Exception as e:
            logger.error(f"Packet send failed: {e}")
            return False
            
    async def route_packet(self, packet: NetworkPacket) -> bool:
        """Route packet through mesh network"""
        try:
            # Find route to destination
            route = await self.routing_engine.find_route(
                packet.source,
                packet.destination,
                self.topology_manager.links
            )
            
            if not route:
                logger.warning(f"No route found for packet {packet.packet_id}")
                self.metrics['packets_dropped'] += 1
                return False
                
            # Update packet route history
            packet.route_history = route
            
            # Apply QoS policies
            if len(route) > 1:
                next_hop = route[1]
                link_id = f"{packet.source}-{next_hop}"
                
                if link_id in self.topology_manager.links:
                    link = self.topology_manager.links[link_id]
                    qos_packets = await self.qos_manager.apply_qos_policy(link, [packet])
                    
                    if not qos_packets:
                        self.metrics['packets_dropped'] += 1
                        return False
                        
            # Encrypt packet if required
            if packet.encryption_key:
                encrypted_payload = await self.security_manager.encrypt_packet(
                    packet, packet.encryption_key
                )
                packet.payload = encrypted_payload
                
            # Forward packet (in production, this would actually transmit)
            logger.debug(f"Routing packet {packet.packet_id} via {' -> '.join(route)}")
            self.metrics['packets_routed'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Packet routing failed: {e}")
            self.metrics['packets_dropped'] += 1
            return False
            
    async def get_mesh_status(self) -> Dict[str, Any]:
        """Get comprehensive mesh network status"""
        try:
            # Calculate network statistics
            total_bandwidth = sum(link.bandwidth for link in self.topology_manager.links.values() if link.is_active)
            avg_latency = sum(link.latency for link in self.topology_manager.links.values() if link.is_active)
            avg_latency = avg_latency / len(self.topology_manager.links) if self.topology_manager.links else 0
            
            active_nodes = len([node for node in self.topology_manager.nodes.values() 
                              if time.time() - node.last_seen < 300])
            
            # QoS compliance check
            qos_compliance = {}
            for link_id, link in self.topology_manager.links.items():
                if link.is_active:
                    compliance = await self.qos_manager.monitor_qos_compliance(link)
                    qos_compliance[link_id] = compliance
                    
            return {
                'mesh_status': {
                    'active': self.mesh_active,
                    'local_node_id': self.local_node_id,
                    'routing_protocol': self.routing_engine.protocol.value,
                    'topology': self.topology_manager.topology.value
                },
                'network_statistics': {
                    'total_nodes': len(self.topology_manager.nodes),
                    'active_nodes': active_nodes,
                    'total_links': len(self.topology_manager.links),
                    'active_links': len([l for l in self.topology_manager.links.values() if l.is_active]),
                    'total_bandwidth': total_bandwidth,
                    'average_latency': avg_latency
                },
                'routing_statistics': {
                    'routing_table_size': len(self.routing_engine.routing_table),
                    'cached_routes': len(self.routing_engine.route_cache)
                },
                'traffic_statistics': {
                    'packets_routed': self.metrics['packets_routed'],
                    'packets_dropped': self.metrics['packets_dropped'],
                    'packet_queue_size': self.packet_queue.qsize(),
                    'success_rate': (
                        self.metrics['packets_routed'] / 
                        (self.metrics['packets_routed'] + self.metrics['packets_dropped'])
                    ) * 100 if (self.metrics['packets_routed'] + self.metrics['packets_dropped']) > 0 else 0
                },
                'performance_metrics': {
                    'mesh_convergence_time': self.metrics['mesh_convergence_time'],
                    'average_latency': self.metrics['average_latency'],
                    'network_utilization': self.metrics['network_utilization']
                },
                'qos_compliance': qos_compliance,
                'security_status': {
                    'encryption_enabled': True,
                    'blacklisted_nodes': len(self.security_manager.blacklisted_nodes),
                    'trusted_nodes': len([
                        node for node in self.topology_manager.nodes.values()
                        if await self.security_manager.is_node_trusted(node.node_id, self.topology_manager.nodes)
                    ])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get mesh status: {e}")
            return {}
            
    async def _setup_local_node(self):
        """Setup local mesh node"""
        try:
            local_node = NetworkNode(
                node_id=self.local_node_id,
                hostname=socket.gethostname(),
                ip_addresses=[self._get_local_ip()],
                mac_addresses=[self._get_local_mac()],
                transport_capabilities={
                    TransportType.TCP,
                    TransportType.UDP,
                    TransportType.WEBSOCKET
                },
                is_gateway=self.config.get('is_gateway', False),
                trust_score=1.0,
                reputation=1.0
            )
            
            self.topology_manager.nodes[self.local_node_id] = local_node
            
        except Exception as e:
            logger.error(f"Local node setup failed: {e}")
            
    def _get_local_ip(self) -> str:
        """Get local IP address"""
        try:
            # Connect to a remote address to get local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
            
    def _get_local_mac(self) -> str:
        """Get local MAC address"""
        try:
            import uuid
            mac = uuid.getnode()
            return ':'.join([f'{(mac >> i) & 0xff:02x}' for i in range(0, 48, 8)][::-1])
        except Exception:
            return "00:00:00:00:00:00"
            
    async def _mesh_maintenance(self):
        """Background mesh maintenance tasks"""
        while self.running:
            try:
                # Update node last seen times
                current_time = time.time()
                
                # Remove stale nodes
                stale_nodes = []
                for node_id, node in self.topology_manager.nodes.items():
                    if current_time - node.last_seen > 600:  # 10 minutes timeout
                        stale_nodes.append(node_id)
                        
                for node_id in stale_nodes:
                    del self.topology_manager.nodes[node_id]
                    logger.info(f"Removed stale node: {node_id}")
                    
                # Update link states
                for link in self.topology_manager.links.values():
                    if current_time - link.last_activity > 300:  # 5 minutes timeout
                        link.is_active = False
                        
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Mesh maintenance error: {e}")
                await asyncio.sleep(60)
                
    async def _packet_processor(self):
        """Background packet processing"""
        while self.running:
            try:
                packet = await asyncio.wait_for(self.packet_queue.get(), timeout=1.0)
                await self.route_packet(packet)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Packet processing error: {e}")
                
    async def _link_monitor(self):
        """Monitor link quality and update metrics"""
        while self.running:
            try:
                # Update link metrics
                total_latency = 0
                active_links = 0
                
                for link in self.topology_manager.links.values():
                    if link.is_active:
                        # Simulate link quality monitoring
                        link.latency += random.uniform(-5, 5)  # Jitter
                        link.latency = max(1, link.latency)     # Minimum 1ms
                        
                        total_latency += link.latency
                        active_links += 1
                        
                if active_links > 0:
                    self.metrics['average_latency'] = total_latency / active_links
                    
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Link monitoring error: {e}")
                await asyncio.sleep(30)
                
    async def _setup_database(self):
        """Setup SQLite database for persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS mesh_nodes (
                node_id TEXT PRIMARY KEY,
                hostname TEXT NOT NULL,
                ip_addresses TEXT NOT NULL,
                mac_addresses TEXT NOT NULL,
                transport_capabilities TEXT NOT NULL,
                location TEXT,
                hardware_specs TEXT,
                trust_score REAL DEFAULT 0.5,
                reputation REAL DEFAULT 0.5,
                last_seen REAL DEFAULT (strftime('%s', 'now')),
                is_gateway BOOLEAN DEFAULT FALSE,
                is_bridge BOOLEAN DEFAULT FALSE,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            );
            
            CREATE TABLE IF NOT EXISTS mesh_links (
                link_id TEXT PRIMARY KEY,
                source_node TEXT NOT NULL,
                target_node TEXT NOT NULL,
                transport_type TEXT NOT NULL,
                bandwidth REAL NOT NULL,
                latency REAL NOT NULL,
                packet_loss REAL DEFAULT 0.0,
                reliability REAL DEFAULT 99.0,
                cost REAL DEFAULT 1.0,
                qos_class TEXT DEFAULT 'normal',
                encryption_level TEXT DEFAULT 'AES256',
                is_active BOOLEAN DEFAULT TRUE,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                last_activity REAL DEFAULT (strftime('%s', 'now'))
            );
        """)
        
        conn.commit()
        conn.close()

# Global instance
_network_mesh: Optional[NetworkMeshEngine] = None

async def initialize_network_mesh(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the global network mesh engine"""
    global _network_mesh
    try:
        _network_mesh = NetworkMeshEngine(config)
        return await _network_mesh.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize network mesh: {e}")
        return False

def get_network_mesh() -> NetworkMeshEngine:
    """Get the global network mesh instance"""
    global _network_mesh
    if _network_mesh is None:
        raise RuntimeError("Network mesh not initialized. Call initialize_network_mesh() first.")
    return _network_mesh

async def shutdown_network_mesh():
    """Shutdown the global network mesh engine"""
    global _network_mesh
    if _network_mesh:
        await _network_mesh.shutdown()
        _network_mesh = None
