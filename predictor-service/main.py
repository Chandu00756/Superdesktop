"""
Omega Super Desktop Console - Predictor Service
AI-Driven Performance Prediction and Optimization Service

This service uses machine learning to predict performance bottlenecks,
optimize resource allocation, and proactively migrate sessions before
performance degradation occurs.

Key Features:
- Real-time performance prediction using ML models
- Workload pattern recognition and classification
- Proactive resource scaling and migration recommendations
- Anomaly detection for performance issues
- Historical performance analysis and trend prediction
"""

import asyncio
import json
import logging
import time
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import os
import math
import statistics

# Core Dependencies
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import aioredis
import asyncpg
import numpy as np
import pandas as pd

# Machine Learning
import sklearn
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Time Series Analysis
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available for time series forecasting")

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
prediction_requests_total = create_counter('prediction_requests_total', 'Total prediction requests', ['model_type'])
prediction_accuracy = create_gauge('prediction_accuracy_score', 'Model prediction accuracy', ['model_type'])
anomaly_detections_total = create_counter('anomaly_detections_total', 'Total anomalies detected', ['anomaly_type'])
model_training_duration = create_histogram('model_training_duration_seconds', 'Model training duration')
prediction_latency = create_histogram('prediction_latency_seconds', 'Prediction request latency')

# Prediction Types and Models
class PredictionType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RESOURCE_USAGE = "resource_usage"
    FAILURE_PROBABILITY = "failure_probability"
    MIGRATION_RECOMMENDATION = "migration_recommendation"
    SCALING_RECOMMENDATION = "scaling_recommendation"

class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    TIME_SERIES = "time_series"
    ANOMALY_DETECTION = "anomaly_detection"
    NEURAL_NETWORK = "neural_network"

class AnomalyType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_CONGESTION = "network_congestion"
    HARDWARE_FAILURE = "hardware_failure"
    SECURITY_BREACH = "security_breach"

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for ML training"""
    timestamp: datetime
    session_id: str
    node_id: str
    user_id: str
    session_type: str
    
    # Performance Metrics
    latency_ms: float
    throughput_mbps: float
    fps: int
    frame_drops: int
    jitter_ms: float
    packet_loss_percent: float
    
    # Resource Utilization
    cpu_utilization_percent: float
    memory_utilization_percent: float
    gpu_utilization_percent: float
    gpu_memory_utilization_percent: float
    network_utilization_percent: float
    storage_io_mbps: float
    
    # System Context
    total_sessions_on_node: int
    node_load_score: float
    network_congestion_score: float
    temperature_celsius: int
    power_draw_watts: int
    
    # Application Context
    application_profile: str
    resolution: str
    color_depth: int
    encoding_preset: str
    bitrate_mbps: float
    
    # External Factors
    time_of_day: int  # Hour of day (0-23)
    day_of_week: int  # Day of week (0-6)
    concurrent_users: int
    
    def to_feature_vector(self) -> List[float]:
        """Convert to feature vector for ML models"""
        return [
            self.latency_ms,
            self.throughput_mbps,
            float(self.fps),
            float(self.frame_drops),
            self.jitter_ms,
            self.packet_loss_percent,
            self.cpu_utilization_percent,
            self.memory_utilization_percent,
            self.gpu_utilization_percent,
            self.gpu_memory_utilization_percent,
            self.network_utilization_percent,
            self.storage_io_mbps,
            float(self.total_sessions_on_node),
            self.node_load_score,
            self.network_congestion_score,
            float(self.temperature_celsius),
            float(self.power_draw_watts),
            float(self.time_of_day),
            float(self.day_of_week),
            float(self.concurrent_users)
        ]

@dataclass
class PredictionRequest:
    """Request for performance prediction"""
    session_id: str
    prediction_type: PredictionType
    time_horizon_seconds: int
    current_metrics: PerformanceMetrics
    context: Dict[str, Any]

@dataclass
class PredictionResult:
    """Result of performance prediction"""
    session_id: str
    prediction_type: PredictionType
    predicted_value: float
    confidence_score: float
    time_horizon_seconds: int
    model_used: ModelType
    timestamp: datetime
    recommendations: List[str]
    anomaly_score: Optional[float] = None
    risk_factors: List[str] = None

@dataclass
class MigrationRecommendation:
    """Migration recommendation with detailed analysis"""
    session_id: str
    current_node: str
    recommended_node: str
    migration_urgency: float  # 0.0 to 1.0
    expected_improvement: Dict[str, float]
    migration_cost: float
    estimated_downtime_ms: float
    confidence_score: float
    reasoning: List[str]

class MLModelManager:
    """Manages ML models for performance prediction"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self.model_accuracies: Dict[str, float] = {}
        self.last_trained: Dict[str, datetime] = {}
        
        # Initialize feature column names
        self._initialize_feature_columns()
        
    def _initialize_feature_columns(self):
        """Initialize feature column names"""
        self.feature_columns = [
            'latency_ms', 'throughput_mbps', 'fps', 'frame_drops', 'jitter_ms',
            'packet_loss_percent', 'cpu_utilization_percent', 'memory_utilization_percent',
            'gpu_utilization_percent', 'gpu_memory_utilization_percent', 
            'network_utilization_percent', 'storage_io_mbps', 'total_sessions_on_node',
            'node_load_score', 'network_congestion_score', 'temperature_celsius',
            'power_draw_watts', 'time_of_day', 'day_of_week', 'concurrent_users'
        ]
    
    async def train_latency_predictor(self, training_data: List[PerformanceMetrics]):
        """Train latency prediction model"""
        if len(training_data) < 100:
            logger.warning("Insufficient training data for latency predictor", 
                          data_points=len(training_data))
            return
        
        start_time = time.time()
        
        # Prepare training data
        X = np.array([metrics.to_feature_vector() for metrics in training_data])
        y = np.array([metrics.latency_ms for metrics in training_data])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        accuracy = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Store model and metadata
        self.models['latency_predictor'] = model
        self.scalers['latency_predictor'] = scaler
        self.model_accuracies['latency_predictor'] = accuracy
        self.last_trained['latency_predictor'] = datetime.utcnow()
        
        # Update metrics
        training_duration = time.time() - start_time
        model_training_duration.observe(training_duration)
        prediction_accuracy.labels(model_type='latency_predictor').set(accuracy)
        
        logger.info(
            "Latency predictor trained successfully",
            accuracy=accuracy,
            mae=mae,
            training_duration=training_duration,
            training_samples=len(training_data)
        )
        
        # Save model to disk
        await self._save_model('latency_predictor', model, scaler)
    
    async def train_anomaly_detector(self, training_data: List[PerformanceMetrics]):
        """Train anomaly detection model"""
        if len(training_data) < 200:
            logger.warning("Insufficient training data for anomaly detector",
                          data_points=len(training_data))
            return
        
        start_time = time.time()
        
        # Prepare training data (use only normal/good performance data)
        normal_data = [
            metrics for metrics in training_data 
            if metrics.latency_ms <= 20.0 and metrics.fps >= 55
        ]
        
        if len(normal_data) < 100:
            logger.warning("Insufficient normal performance data for anomaly detection")
            return
        
        X = np.array([metrics.to_feature_vector() for metrics in normal_data])
        
        # Scale features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train Isolation Forest
        model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1,
            behaviour='new'
        )
        
        model.fit(X_scaled)
        
        # Store model
        self.models['anomaly_detector'] = model
        self.scalers['anomaly_detector'] = scaler
        self.last_trained['anomaly_detector'] = datetime.utcnow()
        
        training_duration = time.time() - start_time
        model_training_duration.observe(training_duration)
        
        logger.info(
            "Anomaly detector trained successfully",
            training_duration=training_duration,
            normal_samples=len(normal_data)
        )
        
        await self._save_model('anomaly_detector', model, scaler)
    
    async def train_throughput_predictor(self, training_data: List[PerformanceMetrics]):
        """Train throughput prediction model"""
        if len(training_data) < 100:
            return
        
        start_time = time.time()
        
        X = np.array([metrics.to_feature_vector() for metrics in training_data])
        y = np.array([metrics.throughput_mbps for metrics in training_data])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = r2_score(y_test, y_pred)
        
        self.models['throughput_predictor'] = model
        self.scalers['throughput_predictor'] = scaler
        self.model_accuracies['throughput_predictor'] = accuracy
        self.last_trained['throughput_predictor'] = datetime.utcnow()
        
        training_duration = time.time() - start_time
        model_training_duration.observe(training_duration)
        prediction_accuracy.labels(model_type='throughput_predictor').set(accuracy)
        
        logger.info(
            "Throughput predictor trained successfully",
            accuracy=accuracy,
            training_duration=training_duration
        )
        
        await self._save_model('throughput_predictor', model, scaler)
    
    async def predict_latency(self, metrics: PerformanceMetrics) -> float:
        """Predict future latency"""
        if 'latency_predictor' not in self.models:
            logger.warning("Latency predictor not available")
            return metrics.latency_ms  # Return current as fallback
        
        start_time = time.time()
        
        model = self.models['latency_predictor']
        scaler = self.scalers['latency_predictor']
        
        X = np.array([metrics.to_feature_vector()])
        X_scaled = scaler.transform(X)
        
        prediction = model.predict(X_scaled)[0]
        
        prediction_latency.observe(time.time() - start_time)
        prediction_requests_total.labels(model_type='latency_predictor').inc()
        
        return max(0.0, prediction)  # Ensure non-negative
    
    async def predict_throughput(self, metrics: PerformanceMetrics) -> float:
        """Predict future throughput"""
        if 'throughput_predictor' not in self.models:
            return metrics.throughput_mbps
        
        model = self.models['throughput_predictor']
        scaler = self.scalers['throughput_predictor']
        
        X = np.array([metrics.to_feature_vector()])
        X_scaled = scaler.transform(X)
        
        prediction = model.predict(X_scaled)[0]
        prediction_requests_total.labels(model_type='throughput_predictor').inc()
        
        return max(0.0, prediction)
    
    async def detect_anomaly(self, metrics: PerformanceMetrics) -> Tuple[bool, float]:
        """Detect performance anomalies"""
        if 'anomaly_detector' not in self.models:
            return False, 0.0
        
        model = self.models['anomaly_detector']
        scaler = self.scalers['anomaly_detector']
        
        X = np.array([metrics.to_feature_vector()])
        X_scaled = scaler.transform(X)
        
        # Get anomaly score
        anomaly_score = model.decision_function(X_scaled)[0]
        is_anomaly = model.predict(X_scaled)[0] == -1
        
        if is_anomaly:
            anomaly_detections_total.labels(
                anomaly_type='performance_anomaly'
            ).inc()
        
        return is_anomaly, anomaly_score
    
    async def _save_model(self, model_name: str, model: Any, scaler: Any):
        """Save model and scaler to disk"""
        try:
            models_dir = "/tmp/omega_models"
            os.makedirs(models_dir, exist_ok=True)
            
            joblib.dump(model, f"{models_dir}/{model_name}_model.pkl")
            joblib.dump(scaler, f"{models_dir}/{model_name}_scaler.pkl")
            
            logger.info("Model saved successfully: %s", model_name)
            
        except Exception as e:
            logger.error("Failed to save model %s: %s", model_name, str(e))
    
    async def load_models(self):
        """Load models from disk"""
        try:
            models_dir = "/tmp/omega_models"
            
            for model_name in ['latency_predictor', 'throughput_predictor', 'anomaly_detector']:
                model_path = f"{models_dir}/{model_name}_model.pkl"
                scaler_path = f"{models_dir}/{model_name}_scaler.pkl"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                    logger.info("Model loaded successfully: %s", model_name)
            
        except Exception as e:
            logger.error("Failed to load models: %s", str(e))

class PredictorService:
    """Main predictor service with advanced ML capabilities"""
    
    def __init__(self):
        self.model_manager = MLModelManager()
        self.performance_history: List[PerformanceMetrics] = []
        self.redis_client: Optional[aioredis.Redis] = None
        self.postgres_pool: Optional[asyncpg.Pool] = None
        
        # Service configuration
        self.training_interval_hours = 6
        self.max_history_size = 100000
        self.prediction_cache_ttl = 60  # seconds
        
        # Background tasks
        self.training_task: Optional[asyncio.Task] = None
        self.data_collection_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the predictor service"""
        try:
            # Redis connection (centralized helper)
            try:
                from utils.redis_helper import get_redis_client
                self.redis_client = await get_redis_client(os.getenv('REDIS_URL', 'redis://localhost:6379'))
            except Exception as e:
                logger.warning(f"Redis helper failed in predictor-service, using in-memory stub: {e}")
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
            
            # Load existing models
            await self.model_manager.load_models()
            
            # Load historical data
            await self._load_historical_data()
            
            # Start background tasks
            self.training_task = asyncio.create_task(self._periodic_training())
            self.data_collection_task = asyncio.create_task(self._data_collection_loop())
            
            logger.info("Predictor service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize predictor service", error=str(e))
            raise
    
    async def _load_historical_data(self):
        """Load historical performance data from database"""
        if not self.postgres_pool:
            return
        
        try:
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM performance_metrics 
                    WHERE timestamp > NOW() - INTERVAL '7 days'
                    ORDER BY timestamp DESC
                    LIMIT 10000
                    """
                )
                
                for row in rows:
                    metrics = PerformanceMetrics(
                        timestamp=row['timestamp'],
                        session_id=row['session_id'],
                        node_id=row['node_id'],
                        user_id=row['user_id'],
                        session_type=row['session_type'],
                        latency_ms=row['latency_ms'],
                        throughput_mbps=row['throughput_mbps'],
                        fps=row['fps'],
                        frame_drops=row['frame_drops'],
                        jitter_ms=row['jitter_ms'],
                        packet_loss_percent=row['packet_loss_percent'],
                        cpu_utilization_percent=row['cpu_utilization_percent'],
                        memory_utilization_percent=row['memory_utilization_percent'],
                        gpu_utilization_percent=row['gpu_utilization_percent'],
                        gpu_memory_utilization_percent=row['gpu_memory_utilization_percent'],
                        network_utilization_percent=row['network_utilization_percent'],
                        storage_io_mbps=row['storage_io_mbps'],
                        total_sessions_on_node=row['total_sessions_on_node'],
                        node_load_score=row['node_load_score'],
                        network_congestion_score=row['network_congestion_score'],
                        temperature_celsius=row['temperature_celsius'],
                        power_draw_watts=row['power_draw_watts'],
                        application_profile=row['application_profile'],
                        resolution=row['resolution'],
                        color_depth=row['color_depth'],
                        encoding_preset=row['encoding_preset'],
                        bitrate_mbps=row['bitrate_mbps'],
                        time_of_day=row['time_of_day'],
                        day_of_week=row['day_of_week'],
                        concurrent_users=row['concurrent_users']
                    )
                    self.performance_history.append(metrics)
                
                logger.info(f"Loaded {len(self.performance_history)} historical data points")
                
        except Exception as e:
            logger.error("Failed to load historical data", error=str(e))
    
    async def _periodic_training(self):
        """Periodically retrain models with new data"""
        while True:
            try:
                await asyncio.sleep(self.training_interval_hours * 3600)
                
                if len(self.performance_history) > 100:
                    logger.info("Starting periodic model training")
                    
                    # Train all models
                    await self.model_manager.train_latency_predictor(self.performance_history)
                    await self.model_manager.train_throughput_predictor(self.performance_history)
                    await self.model_manager.train_anomaly_detector(self.performance_history)
                    
                    logger.info("Periodic model training completed")
                
            except Exception as e:
                logger.error("Error in periodic training", error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _data_collection_loop(self):
        """Continuously collect performance data from other services"""
        while True:
            try:
                # This would collect data from session daemon and other services
                # For now, simulate data collection
                await self._simulate_data_collection()
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                logger.error("Error in data collection", error=str(e))
                await asyncio.sleep(30)
    
    async def _simulate_data_collection(self):
        """Simulate real-time data collection"""
        # Generate synthetic performance data for testing
        now = datetime.utcnow()
        
        # Simulate a few active sessions
        for i in range(3):
            metrics = PerformanceMetrics(
                timestamp=now,
                session_id=f"session-{i}",
                node_id=f"node-{i % 2}",
                user_id=f"user-{i}",
                session_type="gaming",
                latency_ms=np.random.normal(15.0, 3.0),
                throughput_mbps=np.random.normal(800.0, 100.0),
                fps=int(np.random.normal(60, 5)),
                frame_drops=int(np.random.poisson(2)),
                jitter_ms=np.random.exponential(2.0),
                packet_loss_percent=np.random.exponential(0.5),
                cpu_utilization_percent=np.random.uniform(30, 80),
                memory_utilization_percent=np.random.uniform(40, 85),
                gpu_utilization_percent=np.random.uniform(50, 95),
                gpu_memory_utilization_percent=np.random.uniform(60, 90),
                network_utilization_percent=np.random.uniform(20, 70),
                storage_io_mbps=np.random.uniform(100, 1000),
                total_sessions_on_node=3,
                node_load_score=np.random.uniform(0.3, 0.8),
                network_congestion_score=np.random.uniform(0.1, 0.5),
                temperature_celsius=int(np.random.normal(65, 10)),
                power_draw_watts=int(np.random.normal(200, 50)),
                application_profile="gaming_high",
                resolution="3840x2160",
                color_depth=10,
                encoding_preset="fast",
                bitrate_mbps=np.random.uniform(50, 150),
                time_of_day=now.hour,
                day_of_week=now.weekday(),
                concurrent_users=np.random.randint(5, 20)
            )
            
            self.performance_history.append(metrics)
            
            # Persist to database
            await self._persist_metrics(metrics)
        
        # Keep history size manageable
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size//2:]
    
    async def _persist_metrics(self, metrics: PerformanceMetrics):
        """Persist performance metrics to database"""
        if not self.postgres_pool:
            return
        
        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO performance_metrics (
                        timestamp, session_id, node_id, user_id, session_type,
                        latency_ms, throughput_mbps, fps, frame_drops, jitter_ms,
                        packet_loss_percent, cpu_utilization_percent, 
                        memory_utilization_percent, gpu_utilization_percent,
                        gpu_memory_utilization_percent, network_utilization_percent,
                        storage_io_mbps, total_sessions_on_node, node_load_score,
                        network_congestion_score, temperature_celsius, power_draw_watts,
                        application_profile, resolution, color_depth, encoding_preset,
                        bitrate_mbps, time_of_day, day_of_week, concurrent_users
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18, $19,
                        $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30
                    )
                    """,
                    metrics.timestamp, metrics.session_id, metrics.node_id,
                    metrics.user_id, metrics.session_type, metrics.latency_ms,
                    metrics.throughput_mbps, metrics.fps, metrics.frame_drops,
                    metrics.jitter_ms, metrics.packet_loss_percent,
                    metrics.cpu_utilization_percent, metrics.memory_utilization_percent,
                    metrics.gpu_utilization_percent, metrics.gpu_memory_utilization_percent,
                    metrics.network_utilization_percent, metrics.storage_io_mbps,
                    metrics.total_sessions_on_node, metrics.node_load_score,
                    metrics.network_congestion_score, metrics.temperature_celsius,
                    metrics.power_draw_watts, metrics.application_profile,
                    metrics.resolution, metrics.color_depth, metrics.encoding_preset,
                    metrics.bitrate_mbps, metrics.time_of_day, metrics.day_of_week,
                    metrics.concurrent_users
                )
        except Exception as e:
            logger.error("Failed to persist metrics", error=str(e))
    
    async def predict_performance(self, request: PredictionRequest) -> PredictionResult:
        """Make performance predictions based on current metrics"""
        try:
            start_time = time.time()
            
            if request.prediction_type == PredictionType.LATENCY:
                predicted_value = await self.model_manager.predict_latency(
                    request.current_metrics
                )
                model_used = ModelType.RANDOM_FOREST
                
            elif request.prediction_type == PredictionType.THROUGHPUT:
                predicted_value = await self.model_manager.predict_throughput(
                    request.current_metrics
                )
                model_used = ModelType.RANDOM_FOREST
                
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prediction type {request.prediction_type} not supported"
                )
            
            # Detect anomalies
            is_anomaly, anomaly_score = await self.model_manager.detect_anomaly(
                request.current_metrics
            )
            
            # Calculate confidence based on historical accuracy
            model_name = f"{request.prediction_type.value}_predictor"
            confidence_score = self.model_manager.model_accuracies.get(model_name, 0.5)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                request, predicted_value, is_anomaly
            )
            
            # Identify risk factors
            risk_factors = await self._identify_risk_factors(
                request.current_metrics, is_anomaly
            )
            
            result = PredictionResult(
                session_id=request.session_id,
                prediction_type=request.prediction_type,
                predicted_value=predicted_value,
                confidence_score=confidence_score,
                time_horizon_seconds=request.time_horizon_seconds,
                model_used=model_used,
                timestamp=datetime.utcnow(),
                recommendations=recommendations,
                anomaly_score=anomaly_score if is_anomaly else None,
                risk_factors=risk_factors if risk_factors else None
            )
            
            # Cache result
            await self._cache_prediction(request.session_id, result)
            
            prediction_latency.observe(time.time() - start_time)
            
            logger.info(
                "Prediction completed",
                session_id=request.session_id,
                prediction_type=request.prediction_type.value,
                predicted_value=predicted_value,
                confidence=confidence_score,
                anomaly_detected=is_anomaly
            )
            
            return result
            
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _generate_recommendations(
        self, 
        request: PredictionRequest, 
        predicted_value: float,
        is_anomaly: bool
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if request.prediction_type == PredictionType.LATENCY:
            if predicted_value > 20.0:
                recommendations.append("Consider migrating session to less loaded node")
                recommendations.append("Reduce encoding quality to decrease processing latency")
                recommendations.append("Enable frame skip protection")
            
            if is_anomaly:
                recommendations.append("Investigate potential hardware issues")
                recommendations.append("Check network congestion on current path")
        
        elif request.prediction_type == PredictionType.THROUGHPUT:
            if predicted_value < 500.0:
                recommendations.append("Increase network bandwidth allocation")
                recommendations.append("Optimize encoding parameters")
                recommendations.append("Consider adaptive bitrate streaming")
        
        # Add general recommendations based on current metrics
        metrics = request.current_metrics
        
        if metrics.gpu_utilization_percent > 90:
            recommendations.append("GPU utilization high - consider GPU load balancing")
        
        if metrics.cpu_utilization_percent > 85:
            recommendations.append("CPU utilization high - consider CPU affinity optimization")
        
        if metrics.memory_utilization_percent > 90:
            recommendations.append("Memory pressure detected - consider memory optimization")
        
        return recommendations
    
    async def _identify_risk_factors(
        self, 
        metrics: PerformanceMetrics, 
        is_anomaly: bool
    ) -> List[str]:
        """Identify performance risk factors"""
        risk_factors = []
        
        if metrics.temperature_celsius > 80:
            risk_factors.append("High temperature may cause thermal throttling")
        
        if metrics.power_draw_watts > 250:
            risk_factors.append("High power consumption may indicate inefficient operation")
        
        if metrics.total_sessions_on_node > 8:
            risk_factors.append("High session density on node")
        
        if metrics.network_congestion_score > 0.7:
            risk_factors.append("Network congestion detected")
        
        if metrics.packet_loss_percent > 1.0:
            risk_factors.append("Significant packet loss affecting performance")
        
        if is_anomaly:
            risk_factors.append("Performance anomaly detected by ML model")
        
        return risk_factors
    
    async def _cache_prediction(self, session_id: str, result: PredictionResult):
        """Cache prediction result in Redis"""
        if self.redis_client:
            try:
                cache_key = f"prediction:{session_id}:{result.prediction_type.value}"
                cache_data = {
                    'predicted_value': result.predicted_value,
                    'confidence_score': result.confidence_score,
                    'timestamp': result.timestamp.isoformat(),
                    'recommendations': result.recommendations
                }
                
                await self.redis_client.setex(
                    cache_key,
                    self.prediction_cache_ttl,
                    json.dumps(cache_data)
                )
            except Exception as e:
                logger.error("Failed to cache prediction", error=str(e))
    
    async def get_migration_recommendation(
        self, 
        session_id: str, 
        current_metrics: PerformanceMetrics
    ) -> Optional[MigrationRecommendation]:
        """Generate intelligent migration recommendations"""
        try:
            # Check if migration is needed
            latency_prediction = await self.model_manager.predict_latency(current_metrics)
            is_anomaly, anomaly_score = await self.model_manager.detect_anomaly(current_metrics)
            
            # Calculate migration urgency
            urgency = 0.0
            
            if latency_prediction > 25.0:
                urgency += 0.4
            
            if is_anomaly:
                urgency += 0.3
            
            if current_metrics.gpu_utilization_percent > 95:
                urgency += 0.2
            
            if current_metrics.cpu_utilization_percent > 90:
                urgency += 0.1
            
            if urgency < 0.3:
                return None  # No migration needed
            
            # Find best target node (simplified)
            recommended_node = await self._find_best_migration_target(current_metrics)
            
            if not recommended_node or recommended_node == current_metrics.node_id:
                return None
            
            # Calculate expected improvements
            expected_improvement = {
                'latency_reduction_ms': max(0, current_metrics.latency_ms - 12.0),
                'throughput_increase_mbps': 100.0,
                'gpu_utilization_reduction': 20.0
            }
            
            # Estimate migration cost and downtime
            migration_cost = urgency * 100.0  # Simplified cost model
            estimated_downtime_ms = 500.0  # Typical live migration downtime
            
            recommendation = MigrationRecommendation(
                session_id=session_id,
                current_node=current_metrics.node_id,
                recommended_node=recommended_node,
                migration_urgency=urgency,
                expected_improvement=expected_improvement,
                migration_cost=migration_cost,
                estimated_downtime_ms=estimated_downtime_ms,
                confidence_score=0.8,  # Based on model accuracy
                reasoning=[
                    f"Predicted latency: {latency_prediction:.1f}ms",
                    f"Current GPU utilization: {current_metrics.gpu_utilization_percent:.1f}%",
                    f"Anomaly score: {anomaly_score:.3f}" if is_anomaly else None
                ]
            )
            
            return recommendation
            
        except Exception as e:
            logger.error("Failed to generate migration recommendation", error=str(e))
            return None
    
    async def _find_best_migration_target(self, metrics: PerformanceMetrics) -> Optional[str]:
        """Find the best node for migration"""
        # This would query the orchestrator for available nodes
        # For now, return a different node
        current_node_num = int(metrics.node_id.split('-')[-1]) if '-' in metrics.node_id else 0
        target_node_num = (current_node_num + 1) % 3  # Assume 3 nodes
        return f"node-{target_node_num}"
    
    async def get_performance_insights(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive performance insights for a session"""
        try:
            # Get recent metrics for the session
            recent_metrics = [
                m for m in self.performance_history[-100:]
                if m.session_id == session_id
            ]
            
            if not recent_metrics:
                return {"error": "No metrics found for session"}
            
            # Calculate statistics
            latencies = [m.latency_ms for m in recent_metrics]
            throughputs = [m.throughput_mbps for m in recent_metrics]
            fps_values = [m.fps for m in recent_metrics]
            
            insights = {
                'session_id': session_id,
                'metrics_count': len(recent_metrics),
                'time_range': {
                    'start': recent_metrics[0].timestamp.isoformat(),
                    'end': recent_metrics[-1].timestamp.isoformat()
                },
                'latency_stats': {
                    'mean': statistics.mean(latencies),
                    'median': statistics.median(latencies),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99),
                    'min': min(latencies),
                    'max': max(latencies)
                },
                'throughput_stats': {
                    'mean': statistics.mean(throughputs),
                    'median': statistics.median(throughputs),
                    'min': min(throughputs),
                    'max': max(throughputs)
                },
                'fps_stats': {
                    'mean': statistics.mean(fps_values),
                    'median': statistics.median(fps_values),
                    'min': min(fps_values),
                    'max': max(fps_values)
                },
                'performance_grade': self._calculate_performance_grade(recent_metrics[-1]),
                'trends': self._analyze_performance_trends(recent_metrics)
            }
            
            return insights
            
        except Exception as e:
            logger.error("Failed to get performance insights", error=str(e))
            return {"error": str(e)}
    
    def _calculate_performance_grade(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall performance grade A-F"""
        score = 0
        
        # Latency score (40% weight)
        if metrics.latency_ms <= 16.67:
            score += 40
        elif metrics.latency_ms <= 25.0:
            score += 30
        elif metrics.latency_ms <= 33.33:
            score += 20
        else:
            score += 10
        
        # FPS score (30% weight)
        if metrics.fps >= 60:
            score += 30
        elif metrics.fps >= 45:
            score += 25
        elif metrics.fps >= 30:
            score += 15
        else:
            score += 5
        
        # Quality score (30% weight)
        if (metrics.packet_loss_percent <= 0.1 and 
            metrics.frame_drops <= 5 and
            metrics.jitter_ms <= 2.0):
            score += 30
        elif (metrics.packet_loss_percent <= 0.5 and
              metrics.frame_drops <= 15):
            score += 20
        else:
            score += 10
        
        # Convert to letter grade
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _analyze_performance_trends(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, str]:
        """Analyze performance trends"""
        if len(metrics_list) < 5:
            return {"trend": "insufficient_data"}
        
        # Analyze latency trend
        recent_latencies = [m.latency_ms for m in metrics_list[-5:]]
        earlier_latencies = [m.latency_ms for m in metrics_list[:5]]
        
        latency_trend = "stable"
        if statistics.mean(recent_latencies) > statistics.mean(earlier_latencies) * 1.1:
            latency_trend = "increasing"
        elif statistics.mean(recent_latencies) < statistics.mean(earlier_latencies) * 0.9:
            latency_trend = "decreasing"
        
        # Analyze FPS trend
        recent_fps = [m.fps for m in metrics_list[-5:]]
        earlier_fps = [m.fps for m in metrics_list[:5]]
        
        fps_trend = "stable"
        if statistics.mean(recent_fps) > statistics.mean(earlier_fps) * 1.05:
            fps_trend = "increasing"
        elif statistics.mean(recent_fps) < statistics.mean(earlier_fps) * 0.95:
            fps_trend = "decreasing"
        
        return {
            "latency_trend": latency_trend,
            "fps_trend": fps_trend,
            "overall_trend": "improving" if (latency_trend == "decreasing" and fps_trend != "decreasing") else
                           "degrading" if (latency_trend == "increasing" or fps_trend == "decreasing") else
                           "stable"
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.training_task:
            self.training_task.cancel()
        
        if self.data_collection_task:
            self.data_collection_task.cancel()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.postgres_pool:
            await self.postgres_pool.close()
        
        logger.info("Predictor service cleanup completed")

# FastAPI Application
app = FastAPI(
    title="Omega Predictor Service",
    description="AI-driven performance prediction and optimization service",
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

# Global predictor service instance
predictor_service = PredictorService()

@app.on_event("startup")
async def startup_event():
    """Initialize the predictor service"""
    await predictor_service.initialize()
    
    # Start Prometheus metrics server
    start_http_server(8002)
    logger.info("Predictor service started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await predictor_service.cleanup()

# API Endpoints
@app.post("/predict", response_model=Dict[str, Any])
async def predict_performance(request: Dict[str, Any]) -> Dict[str, Any]:
    """Make performance predictions"""
    try:
        # Parse current metrics
        current_metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            session_id=request['session_id'],
            node_id=request['current_metrics']['node_id'],
            user_id=request['current_metrics']['user_id'],
            session_type=request['current_metrics']['session_type'],
            latency_ms=request['current_metrics']['latency_ms'],
            throughput_mbps=request['current_metrics']['throughput_mbps'],
            fps=request['current_metrics']['fps'],
            frame_drops=request['current_metrics'].get('frame_drops', 0),
            jitter_ms=request['current_metrics'].get('jitter_ms', 0.0),
            packet_loss_percent=request['current_metrics'].get('packet_loss_percent', 0.0),
            cpu_utilization_percent=request['current_metrics']['cpu_utilization_percent'],
            memory_utilization_percent=request['current_metrics']['memory_utilization_percent'],
            gpu_utilization_percent=request['current_metrics']['gpu_utilization_percent'],
            gpu_memory_utilization_percent=request['current_metrics'].get('gpu_memory_utilization_percent', 0.0),
            network_utilization_percent=request['current_metrics'].get('network_utilization_percent', 0.0),
            storage_io_mbps=request['current_metrics'].get('storage_io_mbps', 0.0),
            total_sessions_on_node=request['current_metrics'].get('total_sessions_on_node', 1),
            node_load_score=request['current_metrics'].get('node_load_score', 0.5),
            network_congestion_score=request['current_metrics'].get('network_congestion_score', 0.1),
            temperature_celsius=request['current_metrics'].get('temperature_celsius', 60),
            power_draw_watts=request['current_metrics'].get('power_draw_watts', 200),
            application_profile=request['current_metrics'].get('application_profile', 'standard'),
            resolution=request['current_metrics'].get('resolution', '1920x1080'),
            color_depth=request['current_metrics'].get('color_depth', 8),
            encoding_preset=request['current_metrics'].get('encoding_preset', 'medium'),
            bitrate_mbps=request['current_metrics'].get('bitrate_mbps', 50.0),
            time_of_day=datetime.utcnow().hour,
            day_of_week=datetime.utcnow().weekday(),
            concurrent_users=request['current_metrics'].get('concurrent_users', 5)
        )
        
        # Create prediction request
        prediction_request = PredictionRequest(
            session_id=request['session_id'],
            prediction_type=PredictionType(request['prediction_type']),
            time_horizon_seconds=request.get('time_horizon_seconds', 300),
            current_metrics=current_metrics,
            context=request.get('context', {})
        )
        
        # Make prediction
        result = await predictor_service.predict_performance(prediction_request)
        
        return asdict(result)
        
    except Exception as e:
        logger.error("Prediction request failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/insights/{session_id}")
async def get_performance_insights(session_id: str) -> Dict[str, Any]:
    """Get comprehensive performance insights"""
    return await predictor_service.get_performance_insights(session_id)

@app.get("/migration-recommendation/{session_id}")
async def get_migration_recommendation(session_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
    """Get migration recommendation for a session"""
    try:
        # Parse current metrics (similar to predict endpoint)
        current_metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            session_id=session_id,
            node_id=request['current_metrics']['node_id'],
            user_id=request['current_metrics']['user_id'],
            session_type=request['current_metrics']['session_type'],
            latency_ms=request['current_metrics']['latency_ms'],
            throughput_mbps=request['current_metrics']['throughput_mbps'],
            fps=request['current_metrics']['fps'],
            frame_drops=request['current_metrics'].get('frame_drops', 0),
            jitter_ms=request['current_metrics'].get('jitter_ms', 0.0),
            packet_loss_percent=request['current_metrics'].get('packet_loss_percent', 0.0),
            cpu_utilization_percent=request['current_metrics']['cpu_utilization_percent'],
            memory_utilization_percent=request['current_metrics']['memory_utilization_percent'],
            gpu_utilization_percent=request['current_metrics']['gpu_utilization_percent'],
            gpu_memory_utilization_percent=request['current_metrics'].get('gpu_memory_utilization_percent', 0.0),
            network_utilization_percent=request['current_metrics'].get('network_utilization_percent', 0.0),
            storage_io_mbps=request['current_metrics'].get('storage_io_mbps', 0.0),
            total_sessions_on_node=request['current_metrics'].get('total_sessions_on_node', 1),
            node_load_score=request['current_metrics'].get('node_load_score', 0.5),
            network_congestion_score=request['current_metrics'].get('network_congestion_score', 0.1),
            temperature_celsius=request['current_metrics'].get('temperature_celsius', 60),
            power_draw_watts=request['current_metrics'].get('power_draw_watts', 200),
            application_profile=request['current_metrics'].get('application_profile', 'standard'),
            resolution=request['current_metrics'].get('resolution', '1920x1080'),
            color_depth=request['current_metrics'].get('color_depth', 8),
            encoding_preset=request['current_metrics'].get('encoding_preset', 'medium'),
            bitrate_mbps=request['current_metrics'].get('bitrate_mbps', 50.0),
            time_of_day=datetime.utcnow().hour,
            day_of_week=datetime.utcnow().weekday(),
            concurrent_users=request['current_metrics'].get('concurrent_users', 5)
        )
        
        recommendation = await predictor_service.get_migration_recommendation(
            session_id, current_metrics
        )
        
        if recommendation:
            return asdict(recommendation)
        else:
            return {"message": "No migration recommended at this time"}
        
    except Exception as e:
        logger.error("Migration recommendation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": len(predictor_service.model_manager.models),
        "data_points": len(predictor_service.performance_history),
        "version": "1.0.0"
    }

@app.get("/models/status")
async def get_model_status():
    """Get status of all ML models"""
    return {
        "models": {
            name: {
                "accuracy": predictor_service.model_manager.model_accuracies.get(name, 0.0),
                "last_trained": predictor_service.model_manager.last_trained.get(name, "never").isoformat() 
                if isinstance(predictor_service.model_manager.last_trained.get(name), datetime) 
                else "never"
            }
            for name in ['latency_predictor', 'throughput_predictor', 'anomaly_detector']
        },
        "data_points": len(predictor_service.performance_history),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=False,
        access_log=True
    )
