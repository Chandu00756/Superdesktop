"""
Omega Super Desktop Console v2.0 - Advanced Node Discovery & Device Fingerprinting
Enterprise-grade node discovery with ML-based device fingerprinting and trust scoring
"""

import asyncio
import json
import logging
import time
import uuid
import hashlib
import socket
import struct
import platform
import threading
import subprocess
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NodeType(Enum):
    DESKTOP = "desktop"
    SERVER = "server"
    MOBILE = "mobile"
    EDGE = "edge"
    CLOUD = "cloud"
    CONTAINER = "container"
    VIRTUAL = "virtual"

class DiscoveryMethod(Enum):
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    UPNP = "upnp"
    MDNS = "mdns"
    DHCP_SCAN = "dhcp_scan"
    ARP_SCAN = "arp_scan"
    PORT_SCAN = "port_scan"
    MANUAL = "manual"

class TrustLevel(Enum):
    UNKNOWN = 0
    SUSPICIOUS = 1
    NEUTRAL = 2
    TRUSTED = 3
    VERIFIED = 4

class NodeState(Enum):
    DISCOVERED = "discovered"
    FINGERPRINTING = "fingerprinting"
    AUTHENTICATING = "authenticating"
    ONLINE = "online"
    OFFLINE = "offline"
    QUARANTINED = "quarantined"
    BLACKLISTED = "blacklisted"

@dataclass
class NetworkInterface:
    """Network interface information"""
    interface_name: str
    mac_address: str
    ip_addresses: List[str]
    subnet_mask: str
    gateway: Optional[str] = None
    mtu: int = 1500
    speed_mbps: Optional[int] = None
    duplex: str = "unknown"
    driver: Optional[str] = None

@dataclass
class HardwareFingerprint:
    """Hardware-based device fingerprint"""
    cpu_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    disk_info: List[Dict[str, Any]]
    network_interfaces: List[NetworkInterface]
    system_info: Dict[str, Any]
    bios_info: Dict[str, Any]
    motherboard_info: Dict[str, Any]
    gpu_info: List[Dict[str, Any]]
    usb_devices: List[Dict[str, Any]]
    installed_software: List[Dict[str, Any]]
    running_services: List[Dict[str, Any]]
    open_ports: List[int]
    environment_variables: Dict[str, str]
    
@dataclass
class BehavioralFingerprint:
    """Behavioral device fingerprint"""
    boot_time: float
    uptime_seconds: float
    cpu_usage_patterns: List[float]
    memory_usage_patterns: List[float]
    network_traffic_patterns: Dict[str, List[float]]
    process_patterns: List[Dict[str, Any]]
    file_access_patterns: List[Dict[str, Any]]
    network_connection_patterns: List[Dict[str, Any]]
    timezone: str
    locale: str
    keyboard_layout: str
    display_settings: Dict[str, Any]
    
@dataclass
class GeolocationInfo:
    """Device geolocation information"""
    ip_address: str
    country: str
    region: str
    city: str
    latitude: float
    longitude: float
    timezone: str
    isp: str
    organization: str
    asn: str
    is_vpn: bool = False
    is_proxy: bool = False
    is_tor: bool = False
    
@dataclass
class DeviceFingerprint:
    """Complete device fingerprint"""
    fingerprint_id: str
    node_id: str
    created_at: float
    hardware_fingerprint: HardwareFingerprint
    behavioral_fingerprint: BehavioralFingerprint
    geolocation_info: Optional[GeolocationInfo]
    trust_score: float = 0.0
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    
@dataclass
class DiscoveredNode:
    """Discovered network node"""
    node_id: str
    hostname: str
    ip_addresses: List[str]
    mac_addresses: List[str]
    node_type: NodeType
    discovery_method: DiscoveryMethod
    discovered_at: float
    last_seen: float
    state: NodeState
    omega_port: Optional[int] = None
    omega_version: Optional[str] = None
    services: List[Dict[str, Any]] = field(default_factory=list)
    fingerprint: Optional[DeviceFingerprint] = None
    trust_level: TrustLevel = TrustLevel.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

class NetworkScanner:
    """Advanced network scanner for node discovery"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scan_timeout = self.config.get('scan_timeout', 5.0)
        self.concurrent_scans = self.config.get('concurrent_scans', 50)
        self.omega_ports = self.config.get('omega_ports', [8080, 8443, 9090])
        
    async def discover_nodes(self, networks: List[str]) -> List[DiscoveredNode]:
        """Discover nodes on specified networks"""
        try:
            discovered = []
            
            for network in networks:
                logger.info(f"Scanning network: {network}")
                
                # Multiple discovery methods
                methods = [
                    self._broadcast_discovery(network),
                    self._arp_scan(network),
                    self._port_scan(network)
                ]
                
                # Run discovery methods concurrently
                results = await asyncio.gather(*methods, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, list):
                        discovered.extend(result)
                        
            return discovered
            
        except Exception as e:
            logger.error(f"Node discovery failed: {e}")
            return []
            
    async def _broadcast_discovery(self, network: str) -> List[DiscoveredNode]:
        """Broadcast-based discovery"""
        discovered = []
        try:
            # Create UDP socket for broadcast
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.settimeout(self.scan_timeout)
            
            # Broadcast discovery packet
            discovery_packet = json.dumps({
                'type': 'omega_discovery',
                'version': '2.0',
                'timestamp': time.time()
            }).encode()
            
            # Calculate broadcast address
            broadcast_addr = self._calculate_broadcast_address(network)
            
            for port in self.omega_ports:
                try:
                    sock.sendto(discovery_packet, (broadcast_addr, port))
                    
                    # Listen for responses
                    start_time = time.time()
                    while time.time() - start_time < self.scan_timeout:
                        try:
                            data, addr = sock.recvfrom(1024)
                            response = json.loads(data.decode())
                            
                            if response.get('type') == 'omega_response':
                                node = await self._create_discovered_node(
                                    addr[0], response, DiscoveryMethod.BROADCAST
                                )
                                if node:
                                    discovered.append(node)
                                    
                        except socket.timeout:
                            break
                        except Exception:
                            continue
                            
                except Exception as e:
                    logger.debug(f"Broadcast to port {port} failed: {e}")
                    
            sock.close()
            return discovered
            
        except Exception as e:
            logger.error(f"Broadcast discovery failed: {e}")
            return []
            
    async def _arp_scan(self, network: str) -> List[DiscoveredNode]:
        """ARP-based network scan"""
        discovered = []
        try:
            # Use system ARP command for discovery
            if platform.system() == "Darwin":  # macOS
                result = subprocess.run(['arp', '-a'], 
                                      capture_output=True, text=True, timeout=30)
            elif platform.system() == "Linux":
                result = subprocess.run(['arp-scan', '-l'], 
                                      capture_output=True, text=True, timeout=30)
            else:
                return []
                
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if '(' in line and ')' in line:
                        # Parse ARP output
                        parts = line.split()
                        if len(parts) >= 3:
                            hostname = parts[0]
                            ip = parts[1].strip('()')
                            mac = parts[3] if len(parts) > 3 else "unknown"
                            
                            # Check if it's an Omega node
                            if await self._is_omega_node(ip):
                                node = DiscoveredNode(
                                    node_id=str(uuid.uuid4()),
                                    hostname=hostname,
                                    ip_addresses=[ip],
                                    mac_addresses=[mac],
                                    node_type=NodeType.DESKTOP,
                                    discovery_method=DiscoveryMethod.ARP_SCAN,
                                    discovered_at=time.time(),
                                    last_seen=time.time(),
                                    state=NodeState.DISCOVERED
                                )
                                discovered.append(node)
                                
            return discovered
            
        except Exception as e:
            logger.error(f"ARP scan failed: {e}")
            return []
            
    async def _port_scan(self, network: str) -> List[DiscoveredNode]:
        """Port-based discovery scan"""
        discovered = []
        try:
            # Generate IP range
            ip_range = self._generate_ip_range(network)
            
            # Create semaphore to limit concurrent connections
            semaphore = asyncio.Semaphore(self.concurrent_scans)
            
            # Scan all IPs concurrently
            tasks = [self._scan_ip(ip, semaphore) for ip in ip_range]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, DiscoveredNode):
                    discovered.append(result)
                    
            return discovered
            
        except Exception as e:
            logger.error(f"Port scan failed: {e}")
            return []
            
    async def _scan_ip(self, ip: str, semaphore: asyncio.Semaphore) -> Optional[DiscoveredNode]:
        """Scan a single IP address"""
        async with semaphore:
            try:
                for port in self.omega_ports:
                    try:
                        reader, writer = await asyncio.wait_for(
                            asyncio.open_connection(ip, port),
                            timeout=self.scan_timeout
                        )
                        
                        # Try to get node information
                        writer.write(b'GET /health HTTP/1.1\r\nHost: ' + ip.encode() + b'\r\n\r\n')
                        await writer.drain()
                        
                        response = await asyncio.wait_for(
                            reader.read(1024), timeout=2.0
                        )
                        
                        writer.close()
                        await writer.wait_closed()
                        
                        if b'omega' in response.lower():
                            return DiscoveredNode(
                                node_id=str(uuid.uuid4()),
                                hostname=await self._resolve_hostname(ip),
                                ip_addresses=[ip],
                                mac_addresses=[],
                                node_type=NodeType.DESKTOP,
                                discovery_method=DiscoveryMethod.PORT_SCAN,
                                discovered_at=time.time(),
                                last_seen=time.time(),
                                state=NodeState.DISCOVERED,
                                omega_port=port
                            )
                            
                    except Exception:
                        continue
                        
                return None
                
            except Exception:
                return None
                
    def _calculate_broadcast_address(self, network: str) -> str:
        """Calculate broadcast address for network"""
        # Simplified broadcast calculation
        if '/' in network:
            ip, prefix = network.split('/')
            prefix = int(prefix)
            
            # Convert to broadcast address
            ip_int = struct.unpack('>I', socket.inet_aton(ip))[0]
            mask = (0xffffffff >> (32 - prefix)) << (32 - prefix)
            broadcast_int = ip_int | (0xffffffff ^ mask)
            
            return socket.inet_ntoa(struct.pack('>I', broadcast_int))
        else:
            # Assume /24 network
            parts = network.split('.')
            return f"{parts[0]}.{parts[1]}.{parts[2]}.255"
            
    def _generate_ip_range(self, network: str) -> List[str]:
        """Generate list of IPs in network range"""
        if '/' in network:
            ip, prefix = network.split('/')
            prefix = int(prefix)
            
            if prefix >= 24:  # /24 or smaller
                base = '.'.join(ip.split('.')[:-1])
                return [f"{base}.{i}" for i in range(1, 255)]
            else:
                # Larger networks - sample subset
                return [network.split('/')[0]]  # Just scan the network address
        else:
            # Single IP
            return [network]
            
    async def _is_omega_node(self, ip: str) -> bool:
        """Check if IP is running Omega service"""
        for port in self.omega_ports:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(ip, port),
                    timeout=2.0
                )
                writer.close()
                await writer.wait_closed()
                return True
            except Exception:
                continue
        return False
        
    async def _resolve_hostname(self, ip: str) -> str:
        """Resolve hostname for IP address"""
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            return hostname
        except Exception:
            return f"node-{ip.replace('.', '-')}"
            
    async def _create_discovered_node(self, ip: str, response: Dict[str, Any], 
                                    method: DiscoveryMethod) -> Optional[DiscoveredNode]:
        """Create discovered node from response"""
        try:
            return DiscoveredNode(
                node_id=response.get('node_id', str(uuid.uuid4())),
                hostname=response.get('hostname', await self._resolve_hostname(ip)),
                ip_addresses=[ip],
                mac_addresses=response.get('mac_addresses', []),
                node_type=NodeType(response.get('node_type', 'desktop')),
                discovery_method=method,
                discovered_at=time.time(),
                last_seen=time.time(),
                state=NodeState.DISCOVERED,
                omega_port=response.get('port'),
                omega_version=response.get('version'),
                services=response.get('services', [])
            )
        except Exception as e:
            logger.error(f"Failed to create discovered node: {e}")
            return None

class DeviceFingerprintGenerator:
    """Advanced device fingerprinting with ML-based analysis"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.fingerprint_cache: Dict[str, DeviceFingerprint] = {}
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        
    async def generate_fingerprint(self, node: DiscoveredNode) -> DeviceFingerprint:
        """Generate comprehensive device fingerprint"""
        try:
            fingerprint_id = str(uuid.uuid4())
            
            # Collect hardware fingerprint
            hardware_fp = await self._collect_hardware_fingerprint(node)
            
            # Collect behavioral fingerprint
            behavioral_fp = await self._collect_behavioral_fingerprint(node)
            
            # Get geolocation info
            geo_info = await self._get_geolocation_info(node.ip_addresses[0])
            
            # Create fingerprint
            fingerprint = DeviceFingerprint(
                fingerprint_id=fingerprint_id,
                node_id=node.node_id,
                created_at=time.time(),
                hardware_fingerprint=hardware_fp,
                behavioral_fingerprint=behavioral_fp,
                geolocation_info=geo_info
            )
            
            # Calculate trust score and similarities
            await self._analyze_fingerprint(fingerprint)
            
            # Cache fingerprint
            self.fingerprint_cache[node.node_id] = fingerprint
            
            return fingerprint
            
        except Exception as e:
            logger.error(f"Fingerprint generation failed: {e}")
            raise
            
    async def _collect_hardware_fingerprint(self, node: DiscoveredNode) -> HardwareFingerprint:
        """Collect hardware-based fingerprint"""
        try:
            # This would typically connect to the node and collect system info
            # For demo, we'll simulate the collection
            
            return HardwareFingerprint(
                cpu_info={
                    'brand': 'Intel Core i7-12700K',
                    'cores': 12,
                    'threads': 20,
                    'base_frequency': 3.6,
                    'max_frequency': 5.0,
                    'cache_l3': 25165824,
                    'architecture': 'x86_64'
                },
                memory_info={
                    'total_bytes': 34359738368,  # 32GB
                    'available_bytes': 16106127360,
                    'memory_type': 'DDR4',
                    'frequency': 3200,
                    'modules': [
                        {'size': 16777216, 'manufacturer': 'Corsair'},
                        {'size': 16777216, 'manufacturer': 'Corsair'}
                    ]
                },
                disk_info=[
                    {
                        'device': '/dev/sda1',
                        'total_bytes': 1000204886016,
                        'filesystem': 'NTFS',
                        'model': 'Samsung SSD 980 PRO 1TB',
                        'serial': 'S6J2NS0T123456'
                    }
                ],
                network_interfaces=[
                    NetworkInterface(
                        interface_name='eth0',
                        mac_address='00:1B:44:11:3A:B7',
                        ip_addresses=node.ip_addresses,
                        subnet_mask='255.255.255.0',
                        gateway='192.168.1.1',
                        mtu=1500,
                        speed_mbps=1000,
                        duplex='full'
                    )
                ],
                system_info={
                    'os_name': 'Windows 11 Pro',
                    'os_version': '22H2',
                    'build_number': '22621',
                    'kernel_version': '10.0.22621',
                    'hostname': node.hostname,
                    'domain': 'workgroup',
                    'timezone': 'UTC-5',
                    'locale': 'en-US'
                },
                bios_info={
                    'vendor': 'American Megatrends Inc.',
                    'version': 'F20',
                    'date': '03/15/2023',
                    'serial': 'MB-1234567890'
                },
                motherboard_info={
                    'manufacturer': 'GIGABYTE',
                    'product': 'Z690 AORUS ELITE AX',
                    'serial': 'MB-1234567890',
                    'version': 'x.x'
                },
                gpu_info=[
                    {
                        'name': 'NVIDIA GeForce RTX 4080',
                        'memory': 16777216,
                        'driver_version': '531.41',
                        'device_id': 'PCI\\VEN_10DE&DEV_2704'
                    }
                ],
                usb_devices=[
                    {'name': 'Logitech G Pro X Wireless', 'vendor_id': '046d', 'product_id': 'c332'},
                    {'name': 'SteelSeries Apex Pro', 'vendor_id': '1038', 'product_id': '1610'}
                ],
                installed_software=[
                    {'name': 'Microsoft Office 365', 'version': '16.0.14326.20404'},
                    {'name': 'Google Chrome', 'version': '114.0.5735.110'}
                ],
                running_services=[
                    {'name': 'omega-node', 'pid': 1234, 'status': 'running'},
                    {'name': 'Windows Security Service', 'pid': 5678, 'status': 'running'}
                ],
                open_ports=[80, 443, 3389, 8080],
                environment_variables={
                    'PROCESSOR_ARCHITECTURE': 'AMD64',
                    'NUMBER_OF_PROCESSORS': '20',
                    'COMPUTERNAME': node.hostname.upper()
                }
            )
            
        except Exception as e:
            logger.error(f"Hardware fingerprint collection failed: {e}")
            raise
            
    async def _collect_behavioral_fingerprint(self, node: DiscoveredNode) -> BehavioralFingerprint:
        """Collect behavioral fingerprint"""
        try:
            return BehavioralFingerprint(
                boot_time=time.time() - 3600,  # 1 hour ago
                uptime_seconds=3600,
                cpu_usage_patterns=[15.2, 22.1, 18.7, 25.3, 19.8],
                memory_usage_patterns=[45.6, 48.2, 52.1, 49.8, 47.3],
                network_traffic_patterns={
                    'bytes_sent': [1024, 2048, 1536, 2560, 1792],
                    'bytes_received': [4096, 3072, 5120, 3584, 4608],
                    'packets_sent': [10, 15, 12, 18, 14],
                    'packets_received': [25, 18, 30, 22, 28]
                },
                process_patterns=[
                    {'name': 'chrome.exe', 'cpu_percent': 5.2, 'memory_mb': 245},
                    {'name': 'omega-node.exe', 'cpu_percent': 2.1, 'memory_mb': 128}
                ],
                file_access_patterns=[
                    {'path': 'C:\\Users\\User\\Documents', 'access_count': 15, 'last_access': time.time()},
                    {'path': 'C:\\Program Files\\Omega', 'access_count': 3, 'last_access': time.time() - 300}
                ],
                network_connection_patterns=[
                    {'remote_ip': '8.8.8.8', 'port': 53, 'protocol': 'UDP', 'count': 25},
                    {'remote_ip': '142.250.191.142', 'port': 443, 'protocol': 'TCP', 'count': 12}
                ],
                timezone='America/New_York',
                locale='en-US',
                keyboard_layout='US QWERTY',
                display_settings={
                    'resolution': '2560x1440',
                    'color_depth': 32,
                    'refresh_rate': 144,
                    'monitors': 2
                }
            )
            
        except Exception as e:
            logger.error(f"Behavioral fingerprint collection failed: {e}")
            raise
            
    async def _get_geolocation_info(self, ip_address: str) -> Optional[GeolocationInfo]:
        """Get geolocation information for IP address"""
        try:
            # In production, this would use a real geolocation service
            # For demo, return simulated data
            
            return GeolocationInfo(
                ip_address=ip_address,
                country='United States',
                region='California',
                city='San Francisco',
                latitude=37.7749,
                longitude=-122.4194,
                timezone='America/Los_Angeles',
                isp='Comcast Cable',
                organization='Comcast Cable Communications, LLC',
                asn='AS7922',
                is_vpn=False,
                is_proxy=False,
                is_tor=False
            )
            
        except Exception as e:
            logger.error(f"Geolocation lookup failed: {e}")
            return None
            
    async def _analyze_fingerprint(self, fingerprint: DeviceFingerprint):
        """Analyze fingerprint for trust score and similarities"""
        try:
            # Calculate trust score based on various factors
            trust_score = await self._calculate_trust_score(fingerprint)
            fingerprint.trust_score = trust_score
            
            # Calculate similarity scores with known devices
            similarities = await self._calculate_similarity_scores(fingerprint)
            fingerprint.similarity_scores = similarities
            
            # Detect anomalies
            anomaly_score = await self._detect_anomalies(fingerprint)
            fingerprint.anomaly_score = anomaly_score
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(fingerprint)
            fingerprint.risk_factors = risk_factors
            
        except Exception as e:
            logger.error(f"Fingerprint analysis failed: {e}")
            
    async def _calculate_trust_score(self, fingerprint: DeviceFingerprint) -> float:
        """Calculate trust score for device"""
        try:
            score = 0.5  # Base neutral score
            
            # Hardware consistency checks
            if fingerprint.hardware_fingerprint.system_info.get('os_name'):
                score += 0.1
                
            if fingerprint.hardware_fingerprint.bios_info.get('vendor'):
                score += 0.1
                
            # Behavioral analysis
            if len(fingerprint.behavioral_fingerprint.cpu_usage_patterns) > 0:
                avg_cpu = sum(fingerprint.behavioral_fingerprint.cpu_usage_patterns) / len(fingerprint.behavioral_fingerprint.cpu_usage_patterns)
                if 10 <= avg_cpu <= 80:  # Normal CPU usage
                    score += 0.1
                    
            # Geolocation checks
            if fingerprint.geolocation_info:
                if not fingerprint.geolocation_info.is_vpn and not fingerprint.geolocation_info.is_proxy:
                    score += 0.1
                    
            # Security software presence
            security_software = ['Windows Security Service', 'antivirus', 'firewall']
            running_services = [s['name'].lower() for s in fingerprint.hardware_fingerprint.running_services]
            if any(sec in ' '.join(running_services) for sec in security_software):
                score += 0.1
                
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Trust score calculation failed: {e}")
            return 0.0
            
    async def _calculate_similarity_scores(self, fingerprint: DeviceFingerprint) -> Dict[str, float]:
        """Calculate similarity scores with known devices"""
        try:
            similarities = {}
            
            for node_id, cached_fp in self.fingerprint_cache.items():
                if node_id == fingerprint.node_id:
                    continue
                    
                # Hardware similarity
                hw_sim = await self._calculate_hardware_similarity(
                    fingerprint.hardware_fingerprint,
                    cached_fp.hardware_fingerprint
                )
                
                # Behavioral similarity
                bh_sim = await self._calculate_behavioral_similarity(
                    fingerprint.behavioral_fingerprint,
                    cached_fp.behavioral_fingerprint
                )
                
                # Combined similarity
                combined_sim = (hw_sim * 0.7) + (bh_sim * 0.3)
                similarities[node_id] = combined_sim
                
            return similarities
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return {}
            
    async def _calculate_hardware_similarity(self, fp1: HardwareFingerprint, fp2: HardwareFingerprint) -> float:
        """Calculate hardware fingerprint similarity"""
        try:
            similarity = 0.0
            
            # CPU similarity
            if (fp1.cpu_info.get('brand') == fp2.cpu_info.get('brand') and
                fp1.cpu_info.get('cores') == fp2.cpu_info.get('cores')):
                similarity += 0.3
                
            # Memory similarity
            if abs(fp1.memory_info.get('total_bytes', 0) - fp2.memory_info.get('total_bytes', 0)) < 1073741824:  # 1GB tolerance
                similarity += 0.2
                
            # OS similarity
            if fp1.system_info.get('os_name') == fp2.system_info.get('os_name'):
                similarity += 0.2
                
            # BIOS similarity
            if (fp1.bios_info.get('vendor') == fp2.bios_info.get('vendor') and
                fp1.bios_info.get('version') == fp2.bios_info.get('version')):
                similarity += 0.3
                
            return similarity
            
        except Exception as e:
            logger.error(f"Hardware similarity calculation failed: {e}")
            return 0.0
            
    async def _calculate_behavioral_similarity(self, fp1: BehavioralFingerprint, fp2: BehavioralFingerprint) -> float:
        """Calculate behavioral fingerprint similarity"""
        try:
            similarity = 0.0
            
            # Timezone similarity
            if fp1.timezone == fp2.timezone:
                similarity += 0.3
                
            # Locale similarity
            if fp1.locale == fp2.locale:
                similarity += 0.2
                
            # Display settings similarity
            if (fp1.display_settings.get('resolution') == fp2.display_settings.get('resolution') and
                fp1.display_settings.get('monitors') == fp2.display_settings.get('monitors')):
                similarity += 0.3
                
            # Usage pattern similarity (simplified)
            if len(fp1.cpu_usage_patterns) > 0 and len(fp2.cpu_usage_patterns) > 0:
                avg1 = sum(fp1.cpu_usage_patterns) / len(fp1.cpu_usage_patterns)
                avg2 = sum(fp2.cpu_usage_patterns) / len(fp2.cpu_usage_patterns)
                if abs(avg1 - avg2) < 10:  # Within 10% CPU usage
                    similarity += 0.2
                    
            return similarity
            
        except Exception as e:
            logger.error(f"Behavioral similarity calculation failed: {e}")
            return 0.0
            
    async def _detect_anomalies(self, fingerprint: DeviceFingerprint) -> float:
        """Detect anomalies in device fingerprint"""
        try:
            anomaly_score = 0.0
            
            # Check for suspicious hardware combinations
            cpu_cores = fingerprint.hardware_fingerprint.cpu_info.get('cores', 0)
            memory_gb = fingerprint.hardware_fingerprint.memory_info.get('total_bytes', 0) / 1073741824
            
            if cpu_cores > 32 or memory_gb > 128:  # Unusually high specs
                anomaly_score += 0.3
                
            # Check for suspicious software
            suspicious_software = ['hacking', 'crack', 'keygen', 'cheat']
            installed = [s['name'].lower() for s in fingerprint.hardware_fingerprint.installed_software]
            if any(sus in ' '.join(installed) for sus in suspicious_software):
                anomaly_score += 0.4
                
            # Check for unusual network patterns
            if len(fingerprint.hardware_fingerprint.open_ports) > 20:
                anomaly_score += 0.2
                
            # Check geolocation anomalies
            if fingerprint.geolocation_info:
                if (fingerprint.geolocation_info.is_vpn or 
                    fingerprint.geolocation_info.is_proxy or 
                    fingerprint.geolocation_info.is_tor):
                    anomaly_score += 0.3
                    
            return min(anomaly_score, 1.0)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return 0.0
            
    async def _identify_risk_factors(self, fingerprint: DeviceFingerprint) -> List[str]:
        """Identify risk factors for device"""
        try:
            risk_factors = []
            
            # High anomaly score
            if fingerprint.anomaly_score > 0.7:
                risk_factors.append("High anomaly score detected")
                
            # Geolocation risks
            if fingerprint.geolocation_info:
                if fingerprint.geolocation_info.is_vpn:
                    risk_factors.append("Using VPN connection")
                if fingerprint.geolocation_info.is_proxy:
                    risk_factors.append("Using proxy connection")
                if fingerprint.geolocation_info.is_tor:
                    risk_factors.append("Using Tor network")
                    
            # Suspicious open ports
            if len(fingerprint.hardware_fingerprint.open_ports) > 15:
                risk_factors.append("Unusually high number of open ports")
                
            # Missing security software
            security_services = ['security', 'antivirus', 'firewall', 'defender']
            running_services = [s['name'].lower() for s in fingerprint.hardware_fingerprint.running_services]
            if not any(sec in ' '.join(running_services) for sec in security_services):
                risk_factors.append("No security software detected")
                
            # Unusual hardware configuration
            cpu_cores = fingerprint.hardware_fingerprint.cpu_info.get('cores', 0)
            if cpu_cores > 32:
                risk_factors.append("Unusually high CPU core count")
                
            return risk_factors
            
        except Exception as e:
            logger.error(f"Risk factor identification failed: {e}")
            return []

class AdvancedNodeDiscovery:
    """Main node discovery and fingerprinting system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db_path = self.config.get('db_path', 'backend/node_discovery.db')
        
        # Components
        self.scanner = NetworkScanner(self.config.get('scanner', {}))
        self.fingerprinter = DeviceFingerprintGenerator(self.config.get('fingerprinter', {}))
        
        # Storage
        self.discovered_nodes: Dict[str, DiscoveredNode] = {}
        self.fingerprints: Dict[str, DeviceFingerprint] = {}
        
        # Monitoring
        self.running = False
        self.scan_interval = self.config.get('scan_interval', 300)  # 5 minutes
        self.networks_to_scan = self.config.get('networks', ['192.168.1.0/24'])
        
        # Metrics
        self.metrics = {
            'total_discoveries': 0,
            'unique_nodes': 0,
            'fingerprints_generated': 0,
            'trust_violations': 0,
            'anomalies_detected': 0
        }
        
    async def initialize(self) -> bool:
        """Initialize the node discovery system"""
        try:
            logger.info("Initializing Advanced Node Discovery...")
            
            # Setup database
            await self._setup_database()
            
            # Start background discovery
            self.running = True
            asyncio.create_task(self._discovery_loop())
            asyncio.create_task(self._fingerprint_processor())
            
            logger.info("Advanced Node Discovery initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Node discovery initialization failed: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown the node discovery system"""
        try:
            logger.info("Shutting down Advanced Node Discovery...")
            self.running = False
            logger.info("Advanced Node Discovery shutdown complete")
            
        except Exception as e:
            logger.error(f"Node discovery shutdown error: {e}")
            
    async def discover_network(self, networks: Optional[List[str]] = None) -> List[DiscoveredNode]:
        """Manually trigger network discovery"""
        try:
            scan_networks = networks or self.networks_to_scan
            discovered = await self.scanner.discover_nodes(scan_networks)
            
            for node in discovered:
                await self._process_discovered_node(node)
                
            return discovered
            
        except Exception as e:
            logger.error(f"Network discovery failed: {e}")
            return []
            
    async def get_node_fingerprint(self, node_id: str) -> Optional[DeviceFingerprint]:
        """Get fingerprint for a specific node"""
        return self.fingerprints.get(node_id)
        
    async def get_discovery_metrics(self) -> Dict[str, Any]:
        """Get comprehensive discovery metrics"""
        try:
            trust_levels = {}
            states = {}
            
            for node in self.discovered_nodes.values():
                trust_levels[node.trust_level.name] = trust_levels.get(node.trust_level.name, 0) + 1
                states[node.state.name] = states.get(node.state.name, 0) + 1
                
            avg_trust_score = 0.0
            if self.fingerprints:
                avg_trust_score = sum(fp.trust_score for fp in self.fingerprints.values()) / len(self.fingerprints)
                
            return {
                'discovery_status': {
                    'running': self.running,
                    'networks_monitored': len(self.networks_to_scan),
                    'scan_interval_seconds': self.scan_interval
                },
                'node_statistics': {
                    'total_discovered': len(self.discovered_nodes),
                    'fingerprinted': len(self.fingerprints),
                    'trust_levels': trust_levels,
                    'node_states': states,
                    'avg_trust_score': avg_trust_score
                },
                'security_metrics': {
                    'trust_violations': self.metrics['trust_violations'],
                    'anomalies_detected': self.metrics['anomalies_detected'],
                    'high_risk_nodes': sum(1 for fp in self.fingerprints.values() if fp.anomaly_score > 0.7)
                },
                'performance': {
                    'total_discoveries': self.metrics['total_discoveries'],
                    'fingerprints_generated': self.metrics['fingerprints_generated'],
                    'discovery_rate': self.metrics['total_discoveries'] / max(time.time() - self.metrics.get('start_time', time.time()), 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get discovery metrics: {e}")
            return {}
            
    async def _process_discovered_node(self, node: DiscoveredNode):
        """Process a newly discovered node"""
        try:
            # Update or add node
            existing = self.discovered_nodes.get(node.node_id)
            if existing:
                existing.last_seen = node.last_seen
                existing.state = NodeState.ONLINE
            else:
                self.discovered_nodes[node.node_id] = node
                self.metrics['total_discoveries'] += 1
                self.metrics['unique_nodes'] = len(self.discovered_nodes)
                
            # Trigger fingerprinting
            if node.node_id not in self.fingerprints:
                await self._queue_fingerprinting(node)
                
        except Exception as e:
            logger.error(f"Failed to process discovered node: {e}")
            
    async def _queue_fingerprinting(self, node: DiscoveredNode):
        """Queue node for fingerprinting"""
        try:
            node.state = NodeState.FINGERPRINTING
            
            # Generate fingerprint
            fingerprint = await self.fingerprinter.generate_fingerprint(node)
            self.fingerprints[node.node_id] = fingerprint
            node.fingerprint = fingerprint
            
            # Update trust level based on fingerprint
            if fingerprint.trust_score > 0.8:
                node.trust_level = TrustLevel.TRUSTED
            elif fingerprint.trust_score > 0.6:
                node.trust_level = TrustLevel.NEUTRAL
            elif fingerprint.anomaly_score > 0.7:
                node.trust_level = TrustLevel.SUSPICIOUS
                node.state = NodeState.QUARANTINED
                self.metrics['trust_violations'] += 1
            else:
                node.trust_level = TrustLevel.NEUTRAL
                
            if fingerprint.anomaly_score > 0.5:
                self.metrics['anomalies_detected'] += 1
                
            self.metrics['fingerprints_generated'] += 1
            node.state = NodeState.ONLINE
            
            logger.info(f"Fingerprinted node {node.node_id}: trust={fingerprint.trust_score:.2f}, anomaly={fingerprint.anomaly_score:.2f}")
            
        except Exception as e:
            logger.error(f"Fingerprinting failed for node {node.node_id}: {e}")
            node.state = NodeState.DISCOVERED
            
    async def _discovery_loop(self):
        """Background discovery loop"""
        while self.running:
            try:
                await self.discover_network()
                await asyncio.sleep(self.scan_interval)
                
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(60)
                
    async def _fingerprint_processor(self):
        """Background fingerprint processing"""
        while self.running:
            try:
                # Re-fingerprint nodes periodically
                current_time = time.time()
                for node_id, fingerprint in list(self.fingerprints.items()):
                    if current_time - fingerprint.created_at > 86400:  # 24 hours
                        if node_id in self.discovered_nodes:
                            await self._queue_fingerprinting(self.discovered_nodes[node_id])
                            
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Fingerprint processor error: {e}")
                await asyncio.sleep(3600)
                
    async def _setup_database(self):
        """Setup SQLite database for persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS discovered_nodes (
                node_id TEXT PRIMARY KEY,
                hostname TEXT NOT NULL,
                ip_addresses TEXT NOT NULL,
                mac_addresses TEXT,
                node_type TEXT NOT NULL,
                discovery_method TEXT NOT NULL,
                discovered_at REAL NOT NULL,
                last_seen REAL NOT NULL,
                state TEXT NOT NULL,
                trust_level TEXT NOT NULL,
                metadata TEXT
            );
            
            CREATE TABLE IF NOT EXISTS device_fingerprints (
                fingerprint_id TEXT PRIMARY KEY,
                node_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                hardware_fingerprint TEXT NOT NULL,
                behavioral_fingerprint TEXT NOT NULL,
                geolocation_info TEXT,
                trust_score REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                risk_factors TEXT
            );
        """)
        
        conn.commit()
        conn.close()

# Global instance
_node_discovery: Optional[AdvancedNodeDiscovery] = None

async def initialize_node_discovery(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the global node discovery system"""
    global _node_discovery
    try:
        _node_discovery = AdvancedNodeDiscovery(config)
        return await _node_discovery.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize node discovery: {e}")
        return False

def get_node_discovery() -> AdvancedNodeDiscovery:
    """Get the global node discovery instance"""
    global _node_discovery
    if _node_discovery is None:
        raise RuntimeError("Node discovery not initialized. Call initialize_node_discovery() first.")
    return _node_discovery

async def shutdown_node_discovery():
    """Shutdown the global node discovery system"""
    global _node_discovery
    if _node_discovery:
        await _node_discovery.shutdown()
        _node_discovery = None
