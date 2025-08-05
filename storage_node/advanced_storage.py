"""
Omega Super Desktop Console v2.0 - Enhanced Storage Node
Tiered storage node with hot/warm/cold storage, ML-driven optimization,
distributed redundancy, deduplication, and intelligent prefetching
"""

import asyncio
import logging
import time
import json
import uuid
import hashlib
import threading
import sqlite3
import pickle
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import os
import shutil
import psutil
import numpy as np
from pathlib import Path

# ML libraries for prediction and optimization
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Compression and deduplication
import zlib
import lz4.frame
import zstandard as zstd
from fastapi import FastAPI, UploadFile, File
import aiofiles
import aiohttp

logger = logging.getLogger(__name__)

class StorageTier(Enum):
    HOT = "hot"          # NVMe SSD, fastest access
    WARM = "warm"        # SATA SSD, moderate access
    COLD = "cold"        # HDD, slow access
    ARCHIVE = "archive"  # Cloud/tape, very slow access
    MEMORY = "memory"    # RAM cache, ultra-fast

class CompressionType(Enum):
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    ZSTD = "zstd"
    GZIP = "gzip"

class ReplicationLevel(Enum):
    NONE = 0
    SINGLE = 1
    DUAL = 2
    TRIPLE = 3
    QUORUM = 5

@dataclass
class StorageDevice:
    """Represents a physical storage device"""
    device_id: str
    device_path: str
    tier: StorageTier
    total_capacity_gb: float
    available_capacity_gb: float
    read_speed_mbps: float
    write_speed_mbps: float
    iops_read: int
    iops_write: int
    
    # Health and performance
    temperature_celsius: float = 40.0
    wear_level: float = 0.0  # 0.0 to 1.0
    bad_sectors: int = 0
    power_on_hours: int = 0
    
    # Performance characteristics
    latency_ms: float = 1.0
    queue_depth: int = 32
    interface: str = "SATA"  # SATA, NVMe, USB, Network
    
    # Status
    is_healthy: bool = True
    is_mounted: bool = True
    mount_point: str = ""

@dataclass
class StorageBlock:
    """Represents a data block in storage"""
    block_id: str
    file_path: str
    size_bytes: int
    checksum: str
    tier: StorageTier
    compression: CompressionType
    encryption_key: Optional[str] = None
    
    # Access patterns
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    
    # Replication
    replicas: List[str] = field(default_factory=list)
    replication_level: ReplicationLevel = ReplicationLevel.SINGLE
    
    # Metadata
    content_type: str = "application/octet-stream"
    tags: Set[str] = field(default_factory=set)
    user_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccessPattern:
    """Tracks file access patterns for ML prediction"""
    file_id: str
    access_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    access_sizes: deque = field(default_factory=lambda: deque(maxlen=1000))
    sequential_access_ratio: float = 0.0
    read_write_ratio: float = 1.0
    peak_hours: Set[int] = field(default_factory=set)
    user_agents: Set[str] = field(default_factory=set)
    
    # Prediction features
    hourly_access_pattern: List[float] = field(default_factory=lambda: [0.0] * 24)
    weekly_access_pattern: List[float] = field(default_factory=lambda: [0.0] * 7)
    access_velocity: float = 0.0  # Accesses per hour

class MLAccessPredictor:
    """Machine learning model for predicting file access patterns"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'hour_of_day', 'day_of_week', 'file_size_log', 'access_count_log',
            'time_since_last_access', 'sequential_ratio', 'read_write_ratio',
            'avg_access_interval', 'access_velocity', 'file_age_days'
        ]
        
    async def initialize(self):
        """Initialize ML model for access prediction"""
        try:
            if SKLEARN_AVAILABLE:
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    random_state=42,
                    n_jobs=-1
                )
                logger.info("ML Access Predictor initialized with Random Forest")
            else:
                logger.warning("Scikit-learn not available, using heuristic predictor")
                
        except Exception as e:
            logger.error(f"Error initializing ML predictor: {e}")
    
    def extract_features(self, access_pattern: AccessPattern, current_time: float) -> np.ndarray:
        """Extract features from access pattern for prediction"""
        try:
            if not access_pattern.access_times:
                return np.zeros(len(self.feature_columns))
            
            # Time-based features
            current_dt = datetime.fromtimestamp(current_time)
            hour_of_day = current_dt.hour
            day_of_week = current_dt.weekday()
            
            # File access features
            access_times = list(access_pattern.access_times)
            file_size_log = np.log10(max(1, np.mean(access_pattern.access_sizes) or 1))
            access_count_log = np.log10(max(1, len(access_times)))
            
            # Temporal features
            time_since_last_access = current_time - max(access_times) if access_times else 3600
            avg_access_interval = np.mean(np.diff(access_times)) if len(access_times) > 1 else 3600
            file_age_days = (current_time - min(access_times)) / 86400 if access_times else 0
            
            features = np.array([
                hour_of_day,
                day_of_week,
                file_size_log,
                access_count_log,
                time_since_last_access,
                access_pattern.sequential_access_ratio,
                access_pattern.read_write_ratio,
                avg_access_interval,
                access_pattern.access_velocity,
                file_age_days
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros(len(self.feature_columns))
    
    async def train_model(self, access_patterns: Dict[str, AccessPattern]):
        """Train the access prediction model"""
        try:
            if not SKLEARN_AVAILABLE or not access_patterns:
                return
            
            # Prepare training data
            features = []
            targets = []
            current_time = time.time()
            
            for pattern in access_patterns.values():
                if len(pattern.access_times) < 2:
                    continue
                
                feature_vector = self.extract_features(pattern, current_time)
                
                # Target: probability of access in next hour
                recent_accesses = [t for t in pattern.access_times 
                                 if current_time - t < 3600]
                access_probability = min(1.0, len(recent_accesses) / 10.0)
                
                features.append(feature_vector)
                targets.append(access_probability)
            
            if len(features) < 10:
                logger.warning("Insufficient training data for ML model")
                return
            
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            logger.info(f"ML model trained on {len(features)} samples")
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
    
    def predict_access_probability(self, access_pattern: AccessPattern) -> float:
        """Predict probability of file access in next hour"""
        try:
            if not self.is_trained or not SKLEARN_AVAILABLE:
                # Fallback heuristic
                return self._heuristic_prediction(access_pattern)
            
            features = self.extract_features(access_pattern, time.time())
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            probability = self.model.predict(features_scaled)[0]
            return max(0.0, min(1.0, probability))
            
        except Exception as e:
            logger.error(f"Error predicting access: {e}")
            return self._heuristic_prediction(access_pattern)
    
    def _heuristic_prediction(self, access_pattern: AccessPattern) -> float:
        """Heuristic-based access prediction when ML is not available"""
        try:
            current_time = time.time()
            
            if not access_pattern.access_times:
                return 0.1  # Low probability for unaccessed files
            
            # Recent access boost
            last_access = max(access_pattern.access_times)
            time_since_access = current_time - last_access
            
            if time_since_access < 3600:  # Within last hour
                return 0.8
            elif time_since_access < 86400:  # Within last day
                return 0.4
            elif time_since_access < 604800:  # Within last week
                return 0.2
            else:
                return 0.05
                
        except Exception as e:
            logger.error(f"Error in heuristic prediction: {e}")
            return 0.1

class TierManager:
    """Manages storage tier allocation and migration"""
    
    def __init__(self, storage_devices: Dict[str, StorageDevice]):
        self.storage_devices = storage_devices
        self.tier_policies = {}
        self.migration_queue = asyncio.Queue()
        self.migration_stats = {
            'hot_to_warm': 0,
            'warm_to_cold': 0,
            'cold_to_archive': 0,
            'promotions': 0,
            'demotions': 0
        }
        
    async def initialize_tier_policies(self):
        """Initialize tier management policies"""
        try:
            # Define default tier policies
            self.tier_policies = {
                StorageTier.HOT: {
                    'max_file_age_hours': 24,
                    'min_access_frequency': 5.0,  # Accesses per hour
                    'max_capacity_usage': 0.8,
                    'promotion_threshold': 10.0,
                    'demotion_threshold': 1.0
                },
                StorageTier.WARM: {
                    'max_file_age_hours': 168,  # 1 week
                    'min_access_frequency': 1.0,
                    'max_capacity_usage': 0.85,
                    'promotion_threshold': 5.0,
                    'demotion_threshold': 0.1
                },
                StorageTier.COLD: {
                    'max_file_age_hours': 8760,  # 1 year
                    'min_access_frequency': 0.01,
                    'max_capacity_usage': 0.9,
                    'promotion_threshold': 1.0,
                    'demotion_threshold': 0.001
                },
                StorageTier.ARCHIVE: {
                    'max_file_age_hours': float('inf'),
                    'min_access_frequency': 0.0,
                    'max_capacity_usage': 0.95,
                    'promotion_threshold': 0.1,
                    'demotion_threshold': 0.0
                }
            }
            
            logger.info("Tier policies initialized")
            
        except Exception as e:
            logger.error(f"Error initializing tier policies: {e}")
    
    async def optimize_tier_allocation(self, blocks: Dict[str, StorageBlock], 
                                     access_patterns: Dict[str, AccessPattern],
                                     predictor: MLAccessPredictor) -> List[Dict[str, Any]]:
        """Optimize storage tier allocation using ML predictions"""
        try:
            migration_recommendations = []
            current_time = time.time()
            
            for block_id, block in blocks.items():
                if block_id not in access_patterns:
                    continue
                
                pattern = access_patterns[block_id]
                
                # Get access prediction
                access_probability = predictor.predict_access_probability(pattern)
                
                # Determine optimal tier
                optimal_tier = self._determine_optimal_tier(
                    block, pattern, access_probability, current_time
                )
                
                # Check if migration is needed
                if optimal_tier != block.tier:
                    migration_recommendations.append({
                        'block_id': block_id,
                        'current_tier': block.tier.value,
                        'target_tier': optimal_tier.value,
                        'access_probability': access_probability,
                        'file_size_mb': block.size_bytes / (1024 * 1024),
                        'priority': self._calculate_migration_priority(
                            block, pattern, access_probability
                        )
                    })
            
            # Sort by priority
            migration_recommendations.sort(key=lambda x: x['priority'], reverse=True)
            
            logger.info(f"Generated {len(migration_recommendations)} migration recommendations")
            return migration_recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing tier allocation: {e}")
            return []
    
    def _determine_optimal_tier(self, block: StorageBlock, pattern: AccessPattern, 
                              access_probability: float, current_time: float) -> StorageTier:
        """Determine optimal storage tier for a block"""
        try:
            file_age_hours = (current_time - block.creation_time) / 3600
            
            # Hot tier criteria
            if (access_probability > 0.5 or 
                len(pattern.access_times) > 10 or
                file_age_hours < 2):
                return StorageTier.HOT
            
            # Warm tier criteria
            elif (access_probability > 0.1 or
                  len(pattern.access_times) > 3 or
                  file_age_hours < 24):
                return StorageTier.WARM
            
            # Cold tier criteria
            elif (access_probability > 0.01 or
                  len(pattern.access_times) > 0 or
                  file_age_hours < 168):
                return StorageTier.COLD
            
            # Archive tier
            else:
                return StorageTier.ARCHIVE
                
        except Exception as e:
            logger.error(f"Error determining optimal tier: {e}")
            return StorageTier.COLD
    
    def _calculate_migration_priority(self, block: StorageBlock, pattern: AccessPattern, 
                                    access_probability: float) -> float:
        """Calculate migration priority score"""
        try:
            priority = 0.0
            
            # Size factor (larger files get higher priority for demotion)
            size_factor = min(1.0, block.size_bytes / (1024**3))  # Normalize to 1GB
            
            # Access pattern factor
            access_factor = access_probability
            
            # Tier mismatch penalty
            tier_mismatch = abs(self._tier_to_numeric(block.tier) - 
                              self._access_prob_to_tier_numeric(access_probability))
            
            # Calculate composite priority
            priority = (tier_mismatch * 10) + (size_factor * 5) + (access_factor * 3)
            
            return priority
            
        except Exception as e:
            logger.error(f"Error calculating migration priority: {e}")
            return 0.0
    
    def _tier_to_numeric(self, tier: StorageTier) -> int:
        """Convert tier to numeric value for comparison"""
        tier_map = {
            StorageTier.MEMORY: 0,
            StorageTier.HOT: 1,
            StorageTier.WARM: 2,
            StorageTier.COLD: 3,
            StorageTier.ARCHIVE: 4
        }
        return tier_map.get(tier, 3)
    
    def _access_prob_to_tier_numeric(self, probability: float) -> int:
        """Convert access probability to optimal tier numeric"""
        if probability > 0.5:
            return 1  # HOT
        elif probability > 0.1:
            return 2  # WARM
        elif probability > 0.01:
            return 3  # COLD
        else:
            return 4  # ARCHIVE

class DeduplicationEngine:
    """Handles data deduplication and compression"""
    
    def __init__(self):
        self.chunk_hashes: Dict[str, List[str]] = {}  # hash -> list of chunk_ids
        self.compression_stats = {
            'total_original_size': 0,
            'total_compressed_size': 0,
            'deduplication_savings': 0
        }
        
    async def deduplicate_and_compress(self, data: bytes, 
                                     compression_type: CompressionType = CompressionType.ZSTD) -> Tuple[bytes, Dict[str, Any]]:
        """Deduplicate and compress data"""
        try:
            # Calculate content hash
            content_hash = hashlib.sha256(data).hexdigest()
            
            # Check for exact duplicate
            if content_hash in self.chunk_hashes:
                return b'', {
                    'is_duplicate': True,
                    'content_hash': content_hash,
                    'compression_type': 'none',
                    'original_size': len(data),
                    'compressed_size': 0,
                    'compression_ratio': float('inf')
                }
            
            # Compress data
            compressed_data = await self._compress_data(data, compression_type)
            
            # Store hash reference
            self.chunk_hashes[content_hash] = [content_hash]
            
            # Update stats
            self.compression_stats['total_original_size'] += len(data)
            self.compression_stats['total_compressed_size'] += len(compressed_data)
            
            metadata = {
                'is_duplicate': False,
                'content_hash': content_hash,
                'compression_type': compression_type.value,
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'compression_ratio': len(data) / len(compressed_data) if compressed_data else 1.0
            }
            
            return compressed_data, metadata
            
        except Exception as e:
            logger.error(f"Error in deduplication/compression: {e}")
            return data, {'error': str(e)}
    
    async def _compress_data(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified algorithm"""
        try:
            if compression_type == CompressionType.ZLIB:
                return zlib.compress(data, level=6)
            elif compression_type == CompressionType.LZ4:
                return lz4.frame.compress(data)
            elif compression_type == CompressionType.ZSTD:
                cctx = zstd.ZstdCompressor(level=3)
                return cctx.compress(data)
            elif compression_type == CompressionType.GZIP:
                import gzip
                return gzip.compress(data, compresslevel=6)
            else:
                return data  # No compression
                
        except Exception as e:
            logger.error(f"Error compressing data: {e}")
            return data
    
    async def decompress_data(self, compressed_data: bytes, 
                            compression_type: CompressionType) -> bytes:
        """Decompress data"""
        try:
            if compression_type == CompressionType.ZLIB:
                return zlib.decompress(compressed_data)
            elif compression_type == CompressionType.LZ4:
                return lz4.frame.decompress(compressed_data)
            elif compression_type == CompressionType.ZSTD:
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(compressed_data)
            elif compression_type == CompressionType.GZIP:
                import gzip
                return gzip.decompress(compressed_data)
            else:
                return compressed_data  # No compression
                
        except Exception as e:
            logger.error(f"Error decompressing data: {e}")
            return compressed_data

class ReplicationManager:
    """Manages data replication across storage nodes"""
    
    def __init__(self, node_registry: Dict[str, Any]):
        self.node_registry = node_registry
        self.replication_policies = {}
        self.pending_replications = asyncio.Queue()
        
    async def replicate_block(self, block: StorageBlock, target_nodes: List[str]) -> Dict[str, Any]:
        """Replicate block to target nodes"""
        try:
            replication_results = {}
            
            for node_id in target_nodes:
                if node_id not in self.node_registry:
                    continue
                
                node_info = self.node_registry[node_id]
                result = await self._replicate_to_node(block, node_info)
                replication_results[node_id] = result
            
            # Update block replica list
            successful_replicas = [node_id for node_id, result in replication_results.items() 
                                 if result.get('success', False)]
            block.replicas.extend(successful_replicas)
            
            return {
                'success': len(successful_replicas) > 0,
                'replicated_to': successful_replicas,
                'total_replicas': len(block.replicas),
                'results': replication_results
            }
            
        except Exception as e:
            logger.error(f"Error replicating block: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _replicate_to_node(self, block: StorageBlock, node_info: Dict[str, Any]) -> Dict[str, Any]:
        """Replicate block to specific node"""
        try:
            # This would implement actual replication protocol
            # For now, simulate replication
            return {
                'success': True,
                'node_id': node_info.get('node_id'),
                'replica_path': f"{node_info.get('storage_path', '/tmp')}/{block.block_id}"
            }
            
        except Exception as e:
            logger.error(f"Error replicating to node: {e}")
            return {'success': False, 'error': str(e)}

class EnhancedStorageNode:
    """Enhanced Storage Node with intelligent tiering and ML optimization"""
    
    def __init__(self, node_id: str = None, config: Dict[str, Any] = None):
        self.node_id = node_id or f"storage_{uuid.uuid4().hex[:8]}"
        self.config = config or {}
        
        # Storage management
        self.storage_devices: Dict[str, StorageDevice] = {}
        self.storage_blocks: Dict[str, StorageBlock] = {}
        self.access_patterns: Dict[str, AccessPattern] = {}
        
        # Core components
        self.ml_predictor = MLAccessPredictor()
        self.tier_manager = None
        self.deduplication_engine = DeduplicationEngine()
        self.replication_manager = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_reads': 0,
            'total_writes': 0,
            'total_bytes_read': 0,
            'total_bytes_written': 0,
            'average_read_latency_ms': 0.0,
            'average_write_latency_ms': 0.0,
            'cache_hit_ratio': 0.0,
            'compression_ratio': 1.0,
            'deduplication_ratio': 1.0
        }
        
        # Memory cache
        self.memory_cache: Dict[str, bytes] = {}
        self.cache_access_times: Dict[str, float] = {}
        self.max_cache_size_mb = self.config.get('max_cache_size_mb', 1024)
        
        # Database for metadata
        self.metadata_db_path = self.config.get('metadata_db_path', f'{self.node_id}_metadata.db')
        self.metadata_db = None
        
        self.health_status = "initializing"
        
        logger.info(f"Enhanced Storage Node {self.node_id} initialized")
    
    async def initialize(self):
        """Initialize storage node with all capabilities"""
        try:
            self.health_status = "initializing"
            
            # Initialize metadata database
            await self._initialize_metadata_db()
            
            # Detect and initialize storage devices
            await self._detect_storage_devices()
            
            # Initialize ML predictor
            await self.ml_predictor.initialize()
            
            # Initialize tier manager
            self.tier_manager = TierManager(self.storage_devices)
            await self.tier_manager.initialize_tier_policies()
            
            # Initialize replication manager
            self.replication_manager = ReplicationManager({})
            
            # Load existing metadata
            await self._load_metadata()
            
            # Start background services
            await self._start_background_services()
            
            self.health_status = "active"
            logger.info(f"Storage Node {self.node_id} fully initialized")
            
        except Exception as e:
            logger.error(f"Error initializing storage node: {e}")
            self.health_status = "failed"
            raise

# Additional methods and services will continue the implementation...

if __name__ == "__main__":
    storage_node = EnhancedStorageNode()
    asyncio.run(storage_node.initialize())
