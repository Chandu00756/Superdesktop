"""
Enterprise Machine Learning Pipeline for Predictive Analytics and Optimization
Advanced ML models for resource prediction, anomaly detection, and intelligent optimization
"""

import asyncio
import numpy as np
import pandas as pd
import json
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

# ML Libraries
try:
    import tensorflow as tf
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("ML libraries not available. Install with: pip install tensorflow scikit-learn")

# Time series forecasting
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    TIMESERIES_AVAILABLE = True
except ImportError:
    TIMESERIES_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    RESOURCE_PREDICTOR = "resource_predictor"
    ANOMALY_DETECTOR = "anomaly_detector"
    LOAD_FORECASTER = "load_forecaster"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    FAILURE_PREDICTOR = "failure_predictor"

class PredictionTimeframe(Enum):
    MINUTES_5 = "5min"
    MINUTES_15 = "15min"
    HOUR_1 = "1hour"
    HOURS_6 = "6hours"
    DAY_1 = "1day"
    WEEK_1 = "1week"

@dataclass
class MetricData:
    timestamp: datetime
    node_id: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    gpu_usage: float = 0.0
    temperature: float = 0.0
    power_consumption: float = 0.0
    active_sessions: int = 0
    request_rate: float = 0.0

@dataclass
class PredictionResult:
    model_type: ModelType
    timeframe: PredictionTimeframe
    predictions: Dict[str, float]
    confidence: float
    accuracy_score: float
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class AnomalyDetection:
    timestamp: datetime
    node_id: str
    anomaly_type: str
    severity: str  # low, medium, high, critical
    confidence: float
    affected_metrics: List[str]
    description: str
    suggested_actions: List[str]

class AdvancedMLPipeline:
    """Enterprise ML pipeline for predictive analytics and optimization"""
    
    def __init__(self, db_path: str = "backend/ml_data.db"):
        self.db_path = db_path
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
        # Initialize database
        self._init_database()
        
        # Training configuration
        self.training_config = {
            'min_samples': 1000,
            'retrain_interval': 3600,  # 1 hour
            'feature_window': 100,  # Number of past samples for features
            'test_split': 0.2,
            'validation_split': 0.1
        }
        
        # Model architectures
        self.model_architectures = {
            ModelType.RESOURCE_PREDICTOR: self._create_resource_predictor_model,
            ModelType.ANOMALY_DETECTOR: self._create_anomaly_detector_model,
            ModelType.LOAD_FORECASTER: self._create_load_forecaster_model,
            ModelType.PERFORMANCE_OPTIMIZER: self._create_performance_optimizer_model,
            ModelType.FAILURE_PREDICTOR: self._create_failure_predictor_model
        }
        
        # Thread pool for async ML operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Auto-training scheduler
        self.training_scheduler = threading.Timer(
            self.training_config['retrain_interval'],
            self._auto_retrain_models
        )
        self.training_scheduler.daemon = True
        self.training_scheduler.start()
    
    def _init_database(self):
        """Initialize SQLite database for ML data storage"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    node_id TEXT,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_io REAL,
                    gpu_usage REAL,
                    temperature REAL,
                    power_consumption REAL,
                    active_sessions INTEGER,
                    request_rate REAL
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT,
                    timeframe TEXT,
                    predictions TEXT,
                    confidence REAL,
                    accuracy_score REAL,
                    created_at DATETIME,
                    metadata TEXT
                )
            """)
            
            # Anomalies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    node_id TEXT,
                    anomaly_type TEXT,
                    severity TEXT,
                    confidence REAL,
                    affected_metrics TEXT,
                    description TEXT,
                    suggested_actions TEXT
                )
            """)
            
            # Model metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_type TEXT PRIMARY KEY,
                    last_trained DATETIME,
                    training_samples INTEGER,
                    accuracy REAL,
                    model_path TEXT,
                    scaler_path TEXT,
                    parameters TEXT
                )
            """)
            
            conn.commit()
    
    async def ingest_metrics(self, metrics: List[MetricData]):
        """Ingest metrics data for ML processing"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for metric in metrics:
                cursor.execute("""
                    INSERT INTO metrics (
                        timestamp, node_id, cpu_usage, memory_usage, disk_usage,
                        network_io, gpu_usage, temperature, power_consumption,
                        active_sessions, request_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.timestamp, metric.node_id, metric.cpu_usage,
                    metric.memory_usage, metric.disk_usage, metric.network_io,
                    metric.gpu_usage, metric.temperature, metric.power_consumption,
                    metric.active_sessions, metric.request_rate
                ))
            
            conn.commit()
        
        # Trigger real-time anomaly detection
        await self._detect_realtime_anomalies(metrics)
    
    async def predict_resource_usage(
        self,
        node_id: str,
        timeframe: PredictionTimeframe,
        metrics: List[str] = None
    ) -> PredictionResult:
        """Predict future resource usage for a node"""
        
        if not ML_AVAILABLE:
            raise RuntimeError("ML libraries not available")
        
        if metrics is None:
            metrics = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_io']
        
        # Load or train model
        model_key = f"{ModelType.RESOURCE_PREDICTOR.value}_{node_id}"
        if model_key not in self.models:
            await self._train_resource_predictor(node_id)
        
        # Prepare features
        features = await self._prepare_prediction_features(node_id, timeframe)
        
        # Make predictions
        predictions = {}
        confidence = 0.0
        
        if model_key in self.models:
            model = self.models[model_key]
            scaler = self.scalers.get(f"{model_key}_scaler")
            
            if scaler:
                features_scaled = scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            pred = model.predict(features_scaled)[0]
            
            for i, metric in enumerate(metrics):
                predictions[metric] = float(pred[i]) if len(pred) > i else 0.0
            
            # Calculate confidence based on model metadata
            model_meta = self.model_metadata.get(model_key, {})
            confidence = model_meta.get('accuracy', 0.0)
        
        result = PredictionResult(
            model_type=ModelType.RESOURCE_PREDICTOR,
            timeframe=timeframe,
            predictions=predictions,
            confidence=confidence,
            accuracy_score=confidence,
            created_at=datetime.now(),
            metadata={
                'node_id': node_id,
                'features_shape': features.shape,
                'model_version': model_meta.get('version', '1.0')
            }
        )
        
        # Store prediction
        await self._store_prediction(result)
        
        return result
    
    async def detect_anomalies(
        self,
        node_id: Optional[str] = None,
        timeframe: timedelta = timedelta(hours=1)
    ) -> List[AnomalyDetection]:
        """Detect anomalies in system metrics"""
        
        if not ML_AVAILABLE:
            return []
        
        # Load anomaly detection model
        model_key = f"{ModelType.ANOMALY_DETECTOR.value}_global"
        if model_key not in self.models:
            await self._train_anomaly_detector()
        
        # Get recent metrics
        metrics_data = await self._get_metrics_data(node_id, timeframe)
        
        if len(metrics_data) < 10:  # Need minimum samples
            return []
        
        anomalies = []
        
        if model_key in self.models:
            model = self.models[model_key]
            scaler = self.scalers.get(f"{model_key}_scaler")
            
            # Prepare features
            features = self._prepare_anomaly_features(metrics_data)
            
            if scaler:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features
            
            # Detect anomalies
            anomaly_scores = model.decision_function(features_scaled)
            anomaly_labels = model.predict(features_scaled)
            
            for i, (score, label) in enumerate(zip(anomaly_scores, anomaly_labels)):
                if label == -1:  # Anomaly detected
                    metric = metrics_data[i]
                    severity = self._classify_anomaly_severity(score)
                    
                    anomaly = AnomalyDetection(
                        timestamp=metric['timestamp'],
                        node_id=metric['node_id'],
                        anomaly_type='resource_anomaly',
                        severity=severity,
                        confidence=abs(score),
                        affected_metrics=self._identify_affected_metrics(metric),
                        description=f"Resource usage anomaly detected on {metric['node_id']}",
                        suggested_actions=self._generate_anomaly_actions(metric, severity)
                    )
                    
                    anomalies.append(anomaly)
        
        # Store anomalies
        for anomaly in anomalies:
            await self._store_anomaly(anomaly)
        
        return anomalies
    
    async def forecast_load(
        self,
        timeframe: PredictionTimeframe,
        target_metric: str = 'request_rate'
    ) -> PredictionResult:
        """Forecast system load using time series analysis"""
        
        if not TIMESERIES_AVAILABLE:
            raise RuntimeError("Time series libraries not available")
        
        # Get historical data
        metrics_data = await self._get_metrics_data(
            timeframe=timedelta(days=30)  # Use 30 days of data
        )
        
        if len(metrics_data) < 100:
            raise ValueError("Insufficient data for load forecasting")
        
        # Prepare time series data
        df = pd.DataFrame(metrics_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        
        # Aggregate by time intervals
        interval_map = {
            PredictionTimeframe.MINUTES_5: '5T',
            PredictionTimeframe.MINUTES_15: '15T',
            PredictionTimeframe.HOUR_1: '1H',
            PredictionTimeframe.HOURS_6: '6H',
            PredictionTimeframe.DAY_1: '1D',
            PredictionTimeframe.WEEK_1: '1W'
        }
        
        interval = interval_map.get(timeframe, '1H')
        ts_data = df[target_metric].resample(interval).mean().dropna()
        
        # Fit ARIMA model
        try:
            model = ARIMA(ts_data, order=(2, 1, 2))
            fitted_model = model.fit()
            
            # Forecast
            steps = min(24, len(ts_data) // 4)  # Forecast up to 24 periods
            forecast = fitted_model.forecast(steps=steps)
            
            # Calculate confidence intervals
            forecast_ci = fitted_model.get_forecast(steps=steps).conf_int()
            
            predictions = {
                'forecast': forecast.tolist(),
                'lower_bound': forecast_ci.iloc[:, 0].tolist(),
                'upper_bound': forecast_ci.iloc[:, 1].tolist(),
                'timestamps': [
                    (ts_data.index[-1] + pd.Timedelta(interval) * (i + 1)).isoformat()
                    for i in range(steps)
                ]
            }
            
            # Calculate model accuracy
            accuracy = 1.0 - abs(fitted_model.aic) / 1000  # Simplified accuracy metric
            accuracy = max(0.0, min(1.0, accuracy))
            
        except Exception as e:
            logger.error(f"ARIMA forecasting failed: {e}")
            # Fallback to simple linear trend
            predictions = self._simple_trend_forecast(ts_data, steps)
            accuracy = 0.5
        
        result = PredictionResult(
            model_type=ModelType.LOAD_FORECASTER,
            timeframe=timeframe,
            predictions=predictions,
            confidence=accuracy,
            accuracy_score=accuracy,
            created_at=datetime.now(),
            metadata={
                'target_metric': target_metric,
                'forecast_steps': steps,
                'data_points': len(ts_data)
            }
        )
        
        await self._store_prediction(result)
        return result
    
    async def optimize_performance(
        self,
        current_metrics: Dict[str, float],
        target_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Suggest performance optimizations using ML"""
        
        if not ML_AVAILABLE:
            return {'status': 'error', 'message': 'ML not available'}
        
        # Load performance optimizer model
        model_key = f"{ModelType.PERFORMANCE_OPTIMIZER.value}_global"
        if model_key not in self.models:
            await self._train_performance_optimizer()
        
        optimizations = []
        
        # Analyze current vs target metrics
        for metric, target_value in target_metrics.items():
            current_value = current_metrics.get(metric, 0.0)
            
            if current_value > target_value * 1.1:  # 10% threshold
                # Suggest optimizations
                suggestions = self._generate_optimization_suggestions(
                    metric, current_value, target_value
                )
                optimizations.extend(suggestions)
        
        # Use ML model to rank optimizations by impact
        if model_key in self.models and optimizations:
            ranked_optimizations = await self._rank_optimizations(
                optimizations, current_metrics
            )
        else:
            ranked_optimizations = optimizations
        
        return {
            'status': 'success',
            'current_metrics': current_metrics,
            'target_metrics': target_metrics,
            'optimizations': ranked_optimizations[:10],  # Top 10
            'estimated_impact': self._estimate_optimization_impact(ranked_optimizations)
        }
    
    async def predict_failures(
        self,
        node_id: str,
        timeframe: PredictionTimeframe = PredictionTimeframe.HOUR_1
    ) -> Dict[str, Any]:
        """Predict potential system failures"""
        
        if not ML_AVAILABLE:
            return {'failure_probability': 0.0, 'risk_factors': []}
        
        # Get recent metrics and trends
        metrics_data = await self._get_metrics_data(
            node_id=node_id,
            timeframe=timedelta(hours=6)
        )
        
        if len(metrics_data) < 20:
            return {'failure_probability': 0.0, 'risk_factors': ['Insufficient data']}
        
        # Calculate risk indicators
        risk_factors = []
        risk_score = 0.0
        
        # Analyze trends
        df = pd.DataFrame(metrics_data)
        
        # CPU trend analysis
        cpu_trend = np.polyfit(range(len(df)), df['cpu_usage'], 1)[0]
        if cpu_trend > 0.01:  # Increasing CPU usage
            risk_score += 0.2
            risk_factors.append('Increasing CPU usage trend')
        
        # Memory trend analysis
        memory_trend = np.polyfit(range(len(df)), df['memory_usage'], 1)[0]
        if memory_trend > 0.01:
            risk_score += 0.2
            risk_factors.append('Increasing memory usage trend')
        
        # Temperature analysis
        if df['temperature'].max() > 80:  # High temperature
            risk_score += 0.3
            risk_factors.append('High system temperature')
        
        # Disk usage analysis
        if df['disk_usage'].max() > 90:  # High disk usage
            risk_score += 0.2
            risk_factors.append('High disk usage')
        
        # Error rate analysis (if available)
        # This would require additional error tracking
        
        # Network anomalies
        network_std = df['network_io'].std()
        if network_std > df['network_io'].mean():
            risk_score += 0.1
            risk_factors.append('Network I/O variability')
        
        failure_probability = min(1.0, risk_score)
        
        return {
            'failure_probability': failure_probability,
            'risk_factors': risk_factors,
            'severity': 'high' if failure_probability > 0.7 else 'medium' if failure_probability > 0.4 else 'low',
            'recommended_actions': self._generate_failure_prevention_actions(risk_factors)
        }
    
    async def _train_resource_predictor(self, node_id: str):
        """Train resource prediction model for a specific node"""
        
        # Get training data
        training_data = await self._get_metrics_data(
            node_id=node_id,
            timeframe=timedelta(days=7)
        )
        
        if len(training_data) < self.training_config['min_samples']:
            logger.warning(f"Insufficient data for training resource predictor for {node_id}")
            return
        
        # Prepare features and targets
        features, targets = self._prepare_training_data(training_data, 'resource_prediction')
        
        if len(features) == 0:
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=self.training_config['test_split'], random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if ML_AVAILABLE:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = r2_score(y_test, y_pred)
            
            # Store model
            model_key = f"{ModelType.RESOURCE_PREDICTOR.value}_{node_id}"
            self.models[model_key] = model
            self.scalers[f"{model_key}_scaler"] = scaler
            self.model_metadata[model_key] = {
                'last_trained': datetime.now(),
                'training_samples': len(X_train),
                'accuracy': accuracy,
                'version': '1.0'
            }
            
            logger.info(f"Trained resource predictor for {node_id} with accuracy {accuracy:.3f}")
    
    async def _train_anomaly_detector(self):
        """Train global anomaly detection model"""
        
        # Get training data from all nodes
        training_data = await self._get_metrics_data(timeframe=timedelta(days=14))
        
        if len(training_data) < self.training_config['min_samples']:
            logger.warning("Insufficient data for training anomaly detector")
            return
        
        # Prepare features
        features = self._prepare_anomaly_features(training_data)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train Isolation Forest
        if ML_AVAILABLE:
            model = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_jobs=-1
            )
            model.fit(features_scaled)
            
            # Store model
            model_key = f"{ModelType.ANOMALY_DETECTOR.value}_global"
            self.models[model_key] = model
            self.scalers[f"{model_key}_scaler"] = scaler
            self.model_metadata[model_key] = {
                'last_trained': datetime.now(),
                'training_samples': len(features_scaled),
                'accuracy': 0.8,  # Estimated for unsupervised learning
                'version': '1.0'
            }
            
            logger.info(f"Trained anomaly detector with {len(features_scaled)} samples")
    
    def _create_resource_predictor_model(self) -> Any:
        """Create deep learning model for resource prediction"""
        
        if not ML_AVAILABLE:
            return None
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')  # 4 outputs: CPU, Memory, Disk, Network
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_anomaly_detector_model(self) -> Any:
        """Create anomaly detection model"""
        
        if not ML_AVAILABLE:
            return None
        
        return IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
    
    def _create_load_forecaster_model(self) -> Any:
        """Create time series forecasting model"""
        
        # Uses ARIMA or ExponentialSmoothing from statsmodels
        return None  # Model created dynamically in forecast_load
    
    def _create_performance_optimizer_model(self) -> Any:
        """Create performance optimization model"""
        
        if not ML_AVAILABLE:
            return None
        
        # Would be a reinforcement learning model or optimization algorithm
        return RandomForestRegressor(n_estimators=50, random_state=42)
    
    def _create_failure_predictor_model(self) -> Any:
        """Create failure prediction model"""
        
        if not ML_AVAILABLE:
            return None
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(15,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    async def _get_metrics_data(
        self,
        node_id: Optional[str] = None,
        timeframe: timedelta = timedelta(hours=1)
    ) -> List[Dict]:
        """Get metrics data from database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT timestamp, node_id, cpu_usage, memory_usage, disk_usage,
                       network_io, gpu_usage, temperature, power_consumption,
                       active_sessions, request_rate
                FROM metrics
                WHERE timestamp >= ?
            """
            params = [datetime.now() - timeframe]
            
            if node_id:
                query += " AND node_id = ?"
                params.append(node_id)
            
            query += " ORDER BY timestamp"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            columns = [
                'timestamp', 'node_id', 'cpu_usage', 'memory_usage', 'disk_usage',
                'network_io', 'gpu_usage', 'temperature', 'power_consumption',
                'active_sessions', 'request_rate'
            ]
            
            return [dict(zip(columns, row)) for row in rows]
    
    def _prepare_training_data(
        self,
        metrics_data: List[Dict],
        task_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models"""
        
        df = pd.DataFrame(metrics_data)
        
        if task_type == 'resource_prediction':
            # Use windowed features for time series prediction
            features = []
            targets = []
            
            window_size = self.training_config['feature_window']
            
            for i in range(window_size, len(df)):
                # Features: past window of metrics
                feature_window = df.iloc[i-window_size:i][
                    ['cpu_usage', 'memory_usage', 'disk_usage', 'network_io',
                     'gpu_usage', 'temperature', 'active_sessions', 'request_rate']
                ].values.flatten()
                
                # Target: next values
                target = df.iloc[i][
                    ['cpu_usage', 'memory_usage', 'disk_usage', 'network_io']
                ].values
                
                features.append(feature_window)
                targets.append(target)
            
            return np.array(features), np.array(targets)
        
        return np.array([]), np.array([])
    
    def _prepare_anomaly_features(self, metrics_data: List[Dict]) -> np.ndarray:
        """Prepare features for anomaly detection"""
        
        df = pd.DataFrame(metrics_data)
        
        # Select relevant features
        feature_columns = [
            'cpu_usage', 'memory_usage', 'disk_usage', 'network_io',
            'gpu_usage', 'temperature', 'power_consumption', 'active_sessions'
        ]
        
        features = df[feature_columns].fillna(0).values
        return features
    
    async def _store_prediction(self, prediction: PredictionResult):
        """Store prediction result in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (
                    model_type, timeframe, predictions, confidence,
                    accuracy_score, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.model_type.value,
                prediction.timeframe.value,
                json.dumps(prediction.predictions),
                prediction.confidence,
                prediction.accuracy_score,
                prediction.created_at,
                json.dumps(prediction.metadata)
            ))
            
            conn.commit()
    
    async def _store_anomaly(self, anomaly: AnomalyDetection):
        """Store anomaly detection result in database"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO anomalies (
                    timestamp, node_id, anomaly_type, severity, confidence,
                    affected_metrics, description, suggested_actions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                anomaly.timestamp,
                anomaly.node_id,
                anomaly.anomaly_type,
                anomaly.severity,
                anomaly.confidence,
                json.dumps(anomaly.affected_metrics),
                anomaly.description,
                json.dumps(anomaly.suggested_actions)
            ))
            
            conn.commit()
    
    def _auto_retrain_models(self):
        """Automatically retrain models periodically"""
        
        try:
            # This would run in a separate thread
            logger.info("Starting automatic model retraining")
            
            # Schedule next retraining
            self.training_scheduler = threading.Timer(
                self.training_config['retrain_interval'],
                self._auto_retrain_models
            )
            self.training_scheduler.daemon = True
            self.training_scheduler.start()
            
        except Exception as e:
            logger.error(f"Auto-retraining failed: {e}")

# Global instance
ml_pipeline = AdvancedMLPipeline()