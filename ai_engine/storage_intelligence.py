"""
Omega Super Desktop Console v2.0 - Intelligent Storage Management
AI-powered storage optimization with tiered storage, predictive caching, and distributed file systems.
"""

import asyncio
import logging
import time
import json
import os
import hashlib
import mmap
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import threading
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
import aiofiles
import aiohttp
import psutil

# AI/ML imports
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Storage and compression
import lz4.frame
import zstandard as zstd
import sqlite3
import asyncpg
import redis.asyncio as redis

# Monitoring
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Metrics
STORAGE_OPERATIONS = Counter('omega_storage_operations_total', 'Storage operations', ['operation', 'tier'])
STORAGE_LATENCY = Histogram('omega_storage_latency_seconds', 'Storage operation latency', ['operation'])
STORAGE_UTILIZATION = Gauge('omega_storage_utilization_percent', 'Storage utilization', ['tier', 'node'])
CACHE_HIT_RATIO = Gauge('omega_cache_hit_ratio', 'Cache hit ratio', ['cache_type'])

@dataclass
class StorageObject:
    """Storage object metadata"""
    object_id: str
    path: str
    size_bytes: int
    tier: str  # hot, warm, cold, archive
    compression_type: str
    encryption_enabled: bool
    access_count: int
    last_accessed: float
    created_at: float
    checksum: str
    replication_count: int
    predicted_access_time: Optional[float] = None

@dataclass
class StorageTier:
    """Storage tier configuration"""
    name: str
    storage_type: str  # ssd, nvme, hdd, tape, cloud
    max_size_gb: float
    current_size_gb: float
    performance_class: str  # ultra_fast, fast, medium, slow
    cost_per_gb: float
    retention_policy_days: int
    auto_migration_enabled: bool

class IntelligentTieredStorage:
    """AI-powered tiered storage management"""
    
    def __init__(self):
        self.storage_tiers = {}
        self.objects = {}
        self.access_patterns = deque(maxlen=100000)
        self.ml_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.migration_queue = deque()
        self.setup_default_tiers()
        
    def setup_default_tiers(self):
        """Setup default storage tiers"""
        tiers = [
            StorageTier("hot", "nvme", 100.0, 0.0, "ultra_fast", 0.50, 7, True),
            StorageTier("warm", "ssd", 500.0, 0.0, "fast", 0.20, 30, True),
            StorageTier("cold", "hdd", 2000.0, 0.0, "medium", 0.05, 365, True),
            StorageTier("archive", "tape", 10000.0, 0.0, "slow", 0.01, 2555, False)  # 7 years
        ]
        
        for tier in tiers:
            self.storage_tiers[tier.name] = tier
            
        logger.info("Default storage tiers configured")
    
    def add_storage_object(self, object_path: str, tier: str = "hot") -> StorageObject:
        """Add new storage object"""
        try:
            if not os.path.exists(object_path):
                raise FileNotFoundError(f"Object not found: {object_path}")
                
            file_stats = os.stat(object_path)
            object_id = str(uuid.uuid4())
            
            # Calculate checksum
            checksum = self._calculate_checksum(object_path)
            
            storage_obj = StorageObject(
                object_id=object_id,
                path=object_path,
                size_bytes=file_stats.st_size,
                tier=tier,
                compression_type="none",
                encryption_enabled=False,
                access_count=0,
                last_accessed=time.time(),
                created_at=file_stats.st_ctime,
                checksum=checksum,
                replication_count=1
            )
            
            self.objects[object_id] = storage_obj
            
            # Update tier usage
            self.storage_tiers[tier].current_size_gb += file_stats.st_size / (1024**3)
            
            logger.info(f"Storage object added: {object_id} in tier {tier}")
            return storage_obj
            
        except Exception as e:
            logger.error(f"Error adding storage object: {e}")
            raise
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return ""
    
    def record_access(self, object_id: str):
        """Record object access for ML training"""
        if object_id in self.objects:
            obj = self.objects[object_id]
            obj.access_count += 1
            obj.last_accessed = time.time()
            
            # Record access pattern
            access_pattern = {
                'object_id': object_id,
                'timestamp': time.time(),
                'hour_of_day': time.localtime().tm_hour,
                'day_of_week': time.localtime().tm_wday,
                'size_bytes': obj.size_bytes,
                'current_tier': obj.tier,
                'time_since_creation': time.time() - obj.created_at,
                'access_count': obj.access_count
            }
            
            self.access_patterns.append(access_pattern)
            
            # Retrain model periodically
            if len(self.access_patterns) % 1000 == 0:
                self.train_access_prediction_model()
    
    def train_access_prediction_model(self):
        """Train ML model to predict access patterns"""
        try:
            if len(self.access_patterns) < 100:
                logger.warning("Insufficient access data for training")
                return
                
            # Prepare training data
            df = pd.DataFrame(list(self.access_patterns))
            
            # Features for prediction
            features = [
                'hour_of_day', 'day_of_week', 'size_bytes', 
                'time_since_creation', 'access_count'
            ]
            
            X = df[features].values
            
            # Target: will be accessed in next 24 hours (binary classification)
            # For training, we'll predict based on access frequency
            y = (df['access_count'] > df['access_count'].median()).astype(int)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Random Forest classifier
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.ml_model.fit(X_scaled, y)
            self.is_trained = True
            
            logger.info(f"Access prediction model trained on {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error training access prediction model: {e}")
    
    def predict_access_probability(self, object_id: str) -> float:
        """Predict probability of object being accessed in next 24 hours"""
        try:
            if not self.is_trained or object_id not in self.objects:
                return 0.5  # Default probability
                
            obj = self.objects[object_id]
            current_time = time.localtime()
            
            features = np.array([[
                current_time.tm_hour,
                current_time.tm_wday,
                obj.size_bytes,
                time.time() - obj.created_at,
                obj.access_count
            ]])
            
            features_scaled = self.scaler.transform(features)
            probability = self.ml_model.predict_proba(features_scaled)[0][1]
            
            return float(probability)
            
        except Exception as e:
            logger.error(f"Error predicting access probability: {e}")
            return 0.5
    
    def recommend_tier_migration(self, object_id: str) -> Optional[str]:
        """Recommend optimal tier for object based on access patterns"""
        if object_id not in self.objects:
            return None
            
        obj = self.objects[object_id]
        access_probability = self.predict_access_probability(object_id)
        
        # Migration recommendations based on access prediction
        if access_probability > 0.8:
            recommended_tier = "hot"
        elif access_probability > 0.4:
            recommended_tier = "warm"
        elif access_probability > 0.1:
            recommended_tier = "cold"
        else:
            recommended_tier = "archive"
            
        # Don't recommend migration to same tier
        if recommended_tier == obj.tier:
            return None
            
        # Check tier capacity
        target_tier = self.storage_tiers[recommended_tier]
        object_size_gb = obj.size_bytes / (1024**3)
        
        if (target_tier.current_size_gb + object_size_gb) <= target_tier.max_size_gb:
            return recommended_tier
        else:
            return None
    
    async def migrate_object(self, object_id: str, target_tier: str) -> bool:
        """Migrate object to different storage tier"""
        try:
            if object_id not in self.objects:
                return False
                
            obj = self.objects[object_id]
            source_tier = obj.tier
            
            if source_tier == target_tier:
                return True  # No migration needed
                
            # Simulate migration process
            migration_time = self._estimate_migration_time(obj, target_tier)
            
            logger.info(f"Migrating object {object_id} from {source_tier} to {target_tier}")
            
            # Simulate migration delay
            await asyncio.sleep(min(migration_time, 0.1))  # Cap simulation at 100ms
            
            # Update object metadata
            obj.tier = target_tier
            
            # Update tier usage
            object_size_gb = obj.size_bytes / (1024**3)
            self.storage_tiers[source_tier].current_size_gb -= object_size_gb
            self.storage_tiers[target_tier].current_size_gb += object_size_gb
            
            STORAGE_OPERATIONS.labels(operation='migrate', tier=target_tier).inc()
            
            logger.info(f"Migration completed: {object_id} -> {target_tier}")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating object {object_id}: {e}")
            return False
    
    def _estimate_migration_time(self, obj: StorageObject, target_tier: str) -> float:
        """Estimate migration time based on object size and tier performance"""
        source_perf = self.storage_tiers[obj.tier].performance_class
        target_perf = self.storage_tiers[target_tier].performance_class
        
        # Performance class to MB/s mapping
        perf_speeds = {
            "ultra_fast": 5000,  # 5 GB/s
            "fast": 1000,        # 1 GB/s
            "medium": 200,       # 200 MB/s
            "slow": 50           # 50 MB/s
        }
        
        # Use slower of source/target for migration speed
        migration_speed = min(
            perf_speeds.get(source_perf, 100),
            perf_speeds.get(target_perf, 100)
        ) * 1024 * 1024  # Convert to bytes/s
        
        return obj.size_bytes / migration_speed
    
    async def auto_tier_optimization(self):
        """Automatically optimize object placement across tiers"""
        try:
            migrations_performed = 0
            
            for object_id, obj in self.objects.items():
                recommended_tier = self.recommend_tier_migration(object_id)
                
                if recommended_tier:
                    success = await self.migrate_object(object_id, recommended_tier)
                    if success:
                        migrations_performed += 1
                        
                    # Limit migrations per cycle to avoid overwhelming system
                    if migrations_performed >= 10:
                        break
                        
            logger.info(f"Auto-optimization completed: {migrations_performed} migrations")
            return migrations_performed
            
        except Exception as e:
            logger.error(f"Error in auto tier optimization: {e}")
            return 0

class PredictiveCacheManager:
    """AI-powered predictive caching system"""
    
    def __init__(self, max_cache_size_gb: float = 10.0):
        self.max_cache_size_gb = max_cache_size_gb
        self.current_cache_size_gb = 0.0
        self.cache_objects = {}
        self.access_predictor = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.prefetch_queue = deque()
        self.cache_replacement_policy = "ai_optimized"
        
    async def get_cached_object(self, object_key: str) -> Optional[bytes]:
        """Get object from cache"""
        try:
            if object_key in self.cache_objects:
                cache_entry = self.cache_objects[object_key]
                cache_entry['last_accessed'] = time.time()
                cache_entry['access_count'] += 1
                
                self.cache_hits += 1
                CACHE_HIT_RATIO.labels(cache_type='predictive').set(
                    self.cache_hits / (self.cache_hits + self.cache_misses)
                )
                
                return cache_entry['data']
            else:
                self.cache_misses += 1
                CACHE_HIT_RATIO.labels(cache_type='predictive').set(
                    self.cache_hits / (self.cache_hits + self.cache_misses)
                )
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached object: {e}")
            return None
    
    async def cache_object(self, object_key: str, data: bytes, 
                          priority: float = 0.5) -> bool:
        """Cache object with intelligent eviction"""
        try:
            object_size_gb = len(data) / (1024**3)
            
            # Check if object fits in cache
            if object_size_gb > self.max_cache_size_gb:
                logger.warning(f"Object too large for cache: {object_size_gb} GB")
                return False
                
            # Make space if needed
            await self._ensure_cache_space(object_size_gb)
            
            # Add to cache
            cache_entry = {
                'data': data,
                'size_gb': object_size_gb,
                'cached_at': time.time(),
                'last_accessed': time.time(),
                'access_count': 1,
                'priority': priority,
                'predicted_access_time': time.time() + 3600  # Default 1 hour
            }
            
            self.cache_objects[object_key] = cache_entry
            self.current_cache_size_gb += object_size_gb
            
            logger.info(f"Object cached: {object_key} ({object_size_gb:.3f} GB)")
            return True
            
        except Exception as e:
            logger.error(f"Error caching object: {e}")
            return False
    
    async def _ensure_cache_space(self, required_space_gb: float):
        """Ensure sufficient cache space using intelligent eviction"""
        try:
            while (self.current_cache_size_gb + required_space_gb) > self.max_cache_size_gb:
                if not self.cache_objects:
                    break
                    
                # Select victim for eviction
                victim_key = self._select_eviction_victim()
                
                if victim_key:
                    await self._evict_object(victim_key)
                else:
                    break
                    
        except Exception as e:
            logger.error(f"Error ensuring cache space: {e}")
    
    def _select_eviction_victim(self) -> Optional[str]:
        """Select object for cache eviction using AI-enhanced policy"""
        if not self.cache_objects:
            return None
            
        if self.cache_replacement_policy == "ai_optimized":
            return self._ai_optimized_eviction()
        elif self.cache_replacement_policy == "lru":
            return min(self.cache_objects.keys(), 
                      key=lambda k: self.cache_objects[k]['last_accessed'])
        elif self.cache_replacement_policy == "lfu":
            return min(self.cache_objects.keys(),
                      key=lambda k: self.cache_objects[k]['access_count'])
        else:
            return list(self.cache_objects.keys())[0]  # FIFO fallback
    
    def _ai_optimized_eviction(self) -> Optional[str]:
        """AI-optimized cache eviction decision"""
        if not self.cache_objects:
            return None
            
        current_time = time.time()
        scores = {}
        
        for key, entry in self.cache_objects.items():
            # Calculate eviction score (higher = more likely to evict)
            age = current_time - entry['cached_at']
            time_since_access = current_time - entry['last_accessed']
            access_frequency = entry['access_count'] / max(age / 3600, 1)  # accesses per hour
            
            # Predicted future access probability (simplified)
            predicted_access_prob = 1.0 / (1.0 + time_since_access / 3600)
            
            # Combined score (lower is better, more likely to be evicted)
            eviction_score = (
                time_since_access * 0.4 +
                (1.0 / max(access_frequency, 0.1)) * 0.3 +
                (1.0 - predicted_access_prob) * 0.2 +
                (1.0 - entry['priority']) * 0.1
            )
            
            scores[key] = eviction_score
            
        # Return object with highest eviction score
        return max(scores.keys(), key=lambda k: scores[k])
    
    async def _evict_object(self, object_key: str):
        """Evict object from cache"""
        try:
            if object_key in self.cache_objects:
                entry = self.cache_objects[object_key]
                self.current_cache_size_gb -= entry['size_gb']
                del self.cache_objects[object_key]
                
                logger.info(f"Evicted from cache: {object_key}")
                
        except Exception as e:
            logger.error(f"Error evicting object {object_key}: {e}")
    
    async def predictive_prefetch(self, access_patterns: List[Dict[str, Any]]):
        """Predictively prefetch objects based on access patterns"""
        try:
            # Analyze access patterns to predict next likely accesses
            predictions = self._predict_next_accesses(access_patterns)
            
            for object_key, probability in predictions:
                if probability > 0.7 and object_key not in self.cache_objects:
                    # Add to prefetch queue
                    self.prefetch_queue.append({
                        'object_key': object_key,
                        'probability': probability,
                        'timestamp': time.time()
                    })
                    
            # Process prefetch queue (simulate)
            await self._process_prefetch_queue()
            
        except Exception as e:
            logger.error(f"Error in predictive prefetch: {e}")
    
    def _predict_next_accesses(self, access_patterns: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Predict next object accesses based on patterns"""
        # Simplified prediction based on recent access frequency
        object_scores = defaultdict(float)
        recent_time = time.time() - 3600  # Last hour
        
        for pattern in access_patterns:
            if pattern.get('timestamp', 0) > recent_time:
                object_key = pattern.get('object_id', '')
                # Higher score for more recent and frequent accesses
                score = 1.0 / (1.0 + (time.time() - pattern.get('timestamp', 0)) / 3600)
                object_scores[object_key] += score
                
        # Sort by score and return top predictions
        sorted_predictions = sorted(object_scores.items(), key=lambda x: x[1], reverse=True)
        return [(k, min(v, 1.0)) for k, v in sorted_predictions[:10]]
    
    async def _process_prefetch_queue(self):
        """Process prefetch queue and cache predicted objects"""
        try:
            processed = 0
            while self.prefetch_queue and processed < 5:  # Limit prefetches
                item = self.prefetch_queue.popleft()
                
                # Simulate prefetching object
                # In production, this would fetch from storage
                simulated_data = b"prefetched_data"  # Placeholder
                
                await self.cache_object(
                    item['object_key'], 
                    simulated_data, 
                    item['probability']
                )
                
                processed += 1
                
        except Exception as e:
            logger.error(f"Error processing prefetch queue: {e}")

class DistributedFileSystem:
    """Distributed file system with replication and consistency"""
    
    def __init__(self):
        self.nodes = {}
        self.file_metadata = {}
        self.replication_factor = 3
        self.consistency_level = "eventual"  # strong, eventual
        self.chunk_size = 64 * 1024 * 1024  # 64MB chunks
        
    def add_storage_node(self, node_id: str, capacity_gb: float, 
                        node_type: str = "data"):
        """Add storage node to distributed file system"""
        self.nodes[node_id] = {
            'capacity_gb': capacity_gb,
            'used_gb': 0.0,
            'node_type': node_type,
            'status': 'active',
            'last_heartbeat': time.time(),
            'chunks': set()
        }
        logger.info(f"Storage node added: {node_id}")
    
    async def store_file(self, file_path: str, file_data: bytes) -> Dict[str, Any]:
        """Store file in distributed file system"""
        try:
            file_id = str(uuid.uuid4())
            file_size = len(file_data)
            
            # Split file into chunks
            chunks = self._create_chunks(file_data)
            
            # Select storage nodes for each chunk
            chunk_placements = []
            for i, chunk_data in enumerate(chunks):
                chunk_id = f"{file_id}_chunk_{i}"
                nodes = self._select_storage_nodes(len(chunk_data))
                
                # Store chunk on selected nodes
                for node_id in nodes:
                    await self._store_chunk(node_id, chunk_id, chunk_data)
                    self.nodes[node_id]['chunks'].add(chunk_id)
                    
                chunk_placements.append({
                    'chunk_id': chunk_id,
                    'size': len(chunk_data),
                    'nodes': nodes,
                    'checksum': hashlib.sha256(chunk_data).hexdigest()
                })
            
            # Store file metadata
            self.file_metadata[file_id] = {
                'path': file_path,
                'size': file_size,
                'chunks': chunk_placements,
                'replication_factor': self.replication_factor,
                'created_at': time.time(),
                'last_modified': time.time(),
                'access_count': 0
            }
            
            logger.info(f"File stored: {file_id} ({file_size} bytes, {len(chunks)} chunks)")
            
            return {
                'file_id': file_id,
                'size': file_size,
                'chunks': len(chunks),
                'replication_factor': self.replication_factor,
                'storage_nodes': len(set().union(*[cp['nodes'] for cp in chunk_placements]))
            }
            
        except Exception as e:
            logger.error(f"Error storing file: {e}")
            raise
    
    def _create_chunks(self, data: bytes) -> List[bytes]:
        """Split data into chunks"""
        chunks = []
        for i in range(0, len(data), self.chunk_size):
            chunks.append(data[i:i + self.chunk_size])
        return chunks
    
    def _select_storage_nodes(self, chunk_size: int) -> List[str]:
        """Select optimal storage nodes for chunk placement"""
        available_nodes = [
            node_id for node_id, node_info in self.nodes.items()
            if (node_info['status'] == 'active' and 
                (node_info['capacity_gb'] - node_info['used_gb']) * 1024**3 > chunk_size)
        ]
        
        if len(available_nodes) < self.replication_factor:
            logger.warning(f"Insufficient nodes for replication factor {self.replication_factor}")
            return available_nodes
            
        # Select nodes with most available space
        sorted_nodes = sorted(
            available_nodes,
            key=lambda n: self.nodes[n]['capacity_gb'] - self.nodes[n]['used_gb'],
            reverse=True
        )
        
        return sorted_nodes[:self.replication_factor]
    
    async def _store_chunk(self, node_id: str, chunk_id: str, chunk_data: bytes):
        """Store chunk on specific node"""
        try:
            # Simulate chunk storage
            chunk_size_gb = len(chunk_data) / (1024**3)
            self.nodes[node_id]['used_gb'] += chunk_size_gb
            
            # In production, this would use actual network protocols
            await asyncio.sleep(0.001)  # Simulate network latency
            
            STORAGE_OPERATIONS.labels(operation='store', tier='distributed').inc()
            
        except Exception as e:
            logger.error(f"Error storing chunk {chunk_id} on node {node_id}: {e}")
            raise
    
    async def retrieve_file(self, file_id: str) -> Optional[bytes]:
        """Retrieve file from distributed storage"""
        try:
            if file_id not in self.file_metadata:
                return None
                
            metadata = self.file_metadata[file_id]
            chunks_data = []
            
            # Retrieve all chunks
            for chunk_info in metadata['chunks']:
                chunk_data = await self._retrieve_chunk(chunk_info)
                if chunk_data is None:
                    logger.error(f"Failed to retrieve chunk {chunk_info['chunk_id']}")
                    return None
                chunks_data.append(chunk_data)
                
            # Combine chunks
            file_data = b''.join(chunks_data)
            
            # Update access statistics
            metadata['access_count'] += 1
            metadata['last_accessed'] = time.time()
            
            STORAGE_OPERATIONS.labels(operation='retrieve', tier='distributed').inc()
            
            return file_data
            
        except Exception as e:
            logger.error(f"Error retrieving file {file_id}: {e}")
            return None
    
    async def _retrieve_chunk(self, chunk_info: Dict[str, Any]) -> Optional[bytes]:
        """Retrieve chunk from available nodes"""
        chunk_id = chunk_info['chunk_id']
        available_nodes = [
            node_id for node_id in chunk_info['nodes']
            if (node_id in self.nodes and 
                self.nodes[node_id]['status'] == 'active' and
                chunk_id in self.nodes[node_id]['chunks'])
        ]
        
        if not available_nodes:
            return None
            
        # Try nodes in order of preference (could be latency-based)
        for node_id in available_nodes:
            try:
                # Simulate chunk retrieval
                await asyncio.sleep(0.001)  # Simulate network latency
                
                # In production, this would retrieve actual chunk data
                chunk_data = b"simulated_chunk_data"  # Placeholder
                
                # Verify checksum
                if hashlib.sha256(chunk_data).hexdigest() == chunk_info['checksum']:
                    return chunk_data
                else:
                    logger.warning(f"Checksum mismatch for chunk {chunk_id} on node {node_id}")
                    
            except Exception as e:
                logger.error(f"Error retrieving chunk {chunk_id} from node {node_id}: {e}")
                continue
                
        return None

class CompressionManager:
    """Intelligent compression management"""
    
    def __init__(self):
        self.compression_algorithms = {
            'lz4': lz4.frame,
            'zstd': zstd,
            'none': None
        }
        self.compression_stats = defaultdict(dict)
        
    def select_optimal_compression(self, data: bytes, 
                                 compression_level: str = "auto") -> str:
        """Select optimal compression algorithm based on data characteristics"""
        try:
            if compression_level == "auto":
                # Analyze data to select best compression
                return self._analyze_and_select_compression(data)
            else:
                return compression_level
                
        except Exception as e:
            logger.error(f"Error selecting compression: {e}")
            return "lz4"  # Safe fallback
    
    def _analyze_and_select_compression(self, data: bytes) -> str:
        """Analyze data characteristics to select optimal compression"""
        data_size = len(data)
        
        # For small files, use fast compression
        if data_size < 1024 * 1024:  # < 1MB
            return "lz4"
            
        # Sample data to estimate compressibility
        sample_size = min(64 * 1024, data_size)  # 64KB sample
        sample = data[:sample_size]
        
        # Calculate entropy (rough estimate of compressibility)
        entropy = self._calculate_entropy(sample)
        
        # High entropy (less compressible) - use fast compression
        if entropy > 7.5:
            return "lz4"
        # Low entropy (highly compressible) - use better compression
        elif entropy < 6.0:
            return "zstd"
        else:
            return "lz4"  # Balanced choice
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        try:
            if not data:
                return 0.0
                
            byte_counts = defaultdict(int)
            for byte in data:
                byte_counts[byte] += 1
                
            entropy = 0.0
            data_len = len(data)
            
            for count in byte_counts.values():
                probability = count / data_len
                if probability > 0:
                    entropy -= probability * np.log2(probability)
                    
            return entropy
            
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 8.0  # Maximum entropy assumption
    
    async def compress_data(self, data: bytes, algorithm: str = "auto") -> Tuple[bytes, str, float]:
        """Compress data with selected algorithm"""
        try:
            if algorithm == "auto":
                algorithm = self.select_optimal_compression(data)
                
            start_time = time.time()
            
            if algorithm == "lz4":
                compressed = lz4.frame.compress(data)
            elif algorithm == "zstd":
                compressor = zstd.ZstdCompressor(level=3)
                compressed = compressor.compress(data)
            elif algorithm == "none":
                compressed = data
            else:
                raise ValueError(f"Unknown compression algorithm: {algorithm}")
                
            compression_time = time.time() - start_time
            compression_ratio = len(data) / len(compressed) if compressed else 1.0
            
            # Update statistics
            self.compression_stats[algorithm].update({
                'total_size_before': self.compression_stats[algorithm].get('total_size_before', 0) + len(data),
                'total_size_after': self.compression_stats[algorithm].get('total_size_after', 0) + len(compressed),
                'total_time': self.compression_stats[algorithm].get('total_time', 0) + compression_time,
                'operations': self.compression_stats[algorithm].get('operations', 0) + 1
            })
            
            return compressed, algorithm, compression_ratio
            
        except Exception as e:
            logger.error(f"Error compressing data: {e}")
            return data, "none", 1.0
    
    async def decompress_data(self, compressed_data: bytes, algorithm: str) -> bytes:
        """Decompress data"""
        try:
            if algorithm == "lz4":
                return lz4.frame.decompress(compressed_data)
            elif algorithm == "zstd":
                decompressor = zstd.ZstdDecompressor()
                return decompressor.decompress(compressed_data)
            elif algorithm == "none":
                return compressed_data
            else:
                raise ValueError(f"Unknown compression algorithm: {algorithm}")
                
        except Exception as e:
            logger.error(f"Error decompressing data: {e}")
            return compressed_data

class IntelligentStorageManager:
    """Main intelligent storage management system"""
    
    def __init__(self):
        self.tiered_storage = IntelligentTieredStorage()
        self.cache_manager = PredictiveCacheManager()
        self.distributed_fs = DistributedFileSystem()
        self.compression_manager = CompressionManager()
        
        self.storage_status = "initializing"
        self.optimization_enabled = True
        self.performance_metrics = {}
        
        logger.info("Intelligent Storage Manager v2.0 initialized")
    
    async def initialize_storage(self):
        """Initialize all storage components"""
        try:
            self.storage_status = "configuring"
            
            # Setup distributed storage nodes
            self._setup_storage_nodes()
            
            # Start optimization tasks
            asyncio.create_task(self._storage_optimization_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            
            self.storage_status = "active"
            logger.info("Intelligent Storage Manager fully initialized")
            
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
            self.storage_status = "error"
    
    def _setup_storage_nodes(self):
        """Setup default storage nodes"""
        nodes = [
            ("local_nvme", 100.0, "data"),
            ("local_ssd", 500.0, "data"),
            ("local_hdd", 2000.0, "data"),
            ("remote_node_1", 1000.0, "data"),
            ("remote_node_2", 1000.0, "data")
        ]
        
        for node_id, capacity, node_type in nodes:
            self.distributed_fs.add_storage_node(node_id, capacity, node_type)
    
    async def store_object(self, object_path: str, data: Optional[bytes] = None, 
                          tier: str = "auto", compression: str = "auto") -> Dict[str, Any]:
        """Store object with intelligent optimization"""
        try:
            # Read data if not provided
            if data is None:
                async with aiofiles.open(object_path, 'rb') as f:
                    data = await f.read()
            
            # Determine optimal tier if auto
            if tier == "auto":
                tier = self._determine_optimal_tier(object_path, len(data))
            
            # Compress data if beneficial
            compressed_data, compression_algo, compression_ratio = await self.compression_manager.compress_data(
                data, compression
            )
            
            # Store in tiered storage
            storage_obj = self.tiered_storage.add_storage_object(object_path, tier)
            
            # Also store in distributed file system for redundancy
            dfs_result = await self.distributed_fs.store_file(object_path, compressed_data)
            
            # Cache frequently accessed objects
            if tier in ["hot", "warm"]:
                await self.cache_manager.cache_object(storage_obj.object_id, data, 0.8)
            
            result = {
                'object_id': storage_obj.object_id,
                'tier': tier,
                'compression_algorithm': compression_algo,
                'compression_ratio': compression_ratio,
                'original_size': len(data),
                'compressed_size': len(compressed_data),
                'distributed_file_id': dfs_result['file_id'],
                'replication_factor': dfs_result['replication_factor'],
                'storage_nodes': dfs_result['storage_nodes']
            }
            
            logger.info(f"Object stored successfully: {storage_obj.object_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error storing object: {e}")
            raise
    
    def _determine_optimal_tier(self, object_path: str, size_bytes: int) -> str:
        """Determine optimal storage tier based on object characteristics"""
        # File extension analysis
        path_obj = Path(object_path)
        extension = path_obj.suffix.lower()
        
        # Hot tier for frequently accessed file types
        hot_extensions = {'.exe', '.dll', '.so', '.dylib', '.py', '.js', '.html', '.css'}
        if extension in hot_extensions:
            return "hot"
            
        # Warm tier for medium-priority files
        warm_extensions = {'.txt', '.md', '.json', '.xml', '.yml', '.yaml', '.conf'}
        if extension in warm_extensions:
            return "warm"
            
        # Cold tier for large files
        if size_bytes > 100 * 1024 * 1024:  # > 100MB
            return "cold"
            
        # Archive tier for very large files
        if size_bytes > 1024 * 1024 * 1024:  # > 1GB
            return "archive"
            
        return "warm"  # Default tier
    
    async def retrieve_object(self, object_id: str, use_cache: bool = True) -> Optional[bytes]:
        """Retrieve object with intelligent caching"""
        try:
            # Try cache first
            if use_cache:
                cached_data = await self.cache_manager.get_cached_object(object_id)
                if cached_data:
                    logger.info(f"Object retrieved from cache: {object_id}")
                    return cached_data
            
            # Record access for ML training
            self.tiered_storage.record_access(object_id)
            
            # Get from storage
            if object_id in self.tiered_storage.objects:
                storage_obj = self.tiered_storage.objects[object_id]
                
                # Read from file system (simulated)
                async with aiofiles.open(storage_obj.path, 'rb') as f:
                    data = await f.read()
                
                # Cache for future access
                if use_cache:
                    cache_priority = 0.9 if storage_obj.tier == "hot" else 0.5
                    await self.cache_manager.cache_object(object_id, data, cache_priority)
                
                logger.info(f"Object retrieved from storage: {object_id}")
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving object {object_id}: {e}")
            return None
    
    async def _storage_optimization_loop(self):
        """Background storage optimization"""
        while self.optimization_enabled:
            try:
                # Perform tier optimization
                migrations = await self.tiered_storage.auto_tier_optimization()
                
                # Predictive caching
                if len(self.tiered_storage.access_patterns) > 10:
                    await self.cache_manager.predictive_prefetch(
                        list(self.tiered_storage.access_patterns)[-100:]
                    )
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in storage optimization loop: {e}")
                await asyncio.sleep(600)
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring"""
        while self.optimization_enabled:
            try:
                # Update storage utilization metrics
                for tier_name, tier in self.tiered_storage.storage_tiers.items():
                    utilization = (tier.current_size_gb / tier.max_size_gb) * 100
                    STORAGE_UTILIZATION.labels(tier=tier_name, node='local').set(utilization)
                
                # Update cache metrics
                cache_hit_rate = (
                    self.cache_manager.cache_hits / 
                    max(1, self.cache_manager.cache_hits + self.cache_manager.cache_misses)
                )
                CACHE_HIT_RATIO.labels(cache_type='predictive').set(cache_hit_rate)
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        try:
            total_objects = len(self.tiered_storage.objects)
            cache_efficiency = (
                self.cache_manager.cache_hits / 
                max(1, self.cache_manager.cache_hits + self.cache_manager.cache_misses)
            )
            
            # Calculate tier distribution
            tier_distribution = defaultdict(int)
            for obj in self.tiered_storage.objects.values():
                tier_distribution[obj.tier] += 1
            
            self.performance_metrics = {
                'total_objects': total_objects,
                'cache_hit_ratio': cache_efficiency,
                'cache_size_gb': self.cache_manager.current_cache_size_gb,
                'max_cache_size_gb': self.cache_manager.max_cache_size_gb,
                'tier_distribution': dict(tier_distribution),
                'distributed_nodes': len(self.distributed_fs.nodes),
                'active_nodes': sum(1 for n in self.distributed_fs.nodes.values() if n['status'] == 'active'),
                'compression_stats': dict(self.compression_manager.compression_stats),
                'ml_model_trained': self.tiered_storage.is_trained
            }
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_storage_intelligence_status(self) -> Dict[str, Any]:
        """Get comprehensive storage intelligence status"""
        return {
            'engine_version': '2.0.0',
            'status': self.storage_status,
            'components': {
                'tiered_storage': {
                    'status': 'active',
                    'tiers': len(self.tiered_storage.storage_tiers),
                    'total_objects': len(self.tiered_storage.objects),
                    'ml_model_trained': self.tiered_storage.is_trained
                },
                'predictive_cache': {
                    'status': 'active',
                    'hit_ratio': self.cache_manager.cache_hits / max(1, self.cache_manager.cache_hits + self.cache_manager.cache_misses),
                    'utilization': (self.cache_manager.current_cache_size_gb / self.cache_manager.max_cache_size_gb) * 100,
                    'cached_objects': len(self.cache_manager.cache_objects)
                },
                'distributed_filesystem': {
                    'status': 'active',
                    'total_nodes': len(self.distributed_fs.nodes),
                    'active_nodes': sum(1 for n in self.distributed_fs.nodes.values() if n['status'] == 'active'),
                    'replication_factor': self.distributed_fs.replication_factor
                },
                'compression_manager': {
                    'status': 'active',
                    'algorithms': list(self.compression_manager.compression_algorithms.keys()),
                    'statistics': dict(self.compression_manager.compression_stats)
                }
            },
            'performance_metrics': self.performance_metrics,
            'capabilities': [
                'AI-driven tier optimization',
                'Predictive caching and prefetching',
                'Distributed file storage with replication',
                'Intelligent compression selection',
                'Real-time performance monitoring',
                'Automated storage optimization'
            ]
        }

# Global Intelligent Storage Manager instance
storage_intelligence = IntelligentStorageManager()
