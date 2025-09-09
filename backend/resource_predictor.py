"""
Omega Super Desktop Console v2.0 - Resource Prediction Engine
Production-grade ML-powered resource prediction with time series analysis,
workload pattern recognition, and capacity planning capabilities.
"""

import asyncio
import logging
import time
import json
import math
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

logger = logging.getLogger(__name__)

class MetricType(Enum):
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    NETWORK_THROUGHPUT = "network_throughput"
    STORAGE_IOPS = "storage_iops"
    ENERGY_CONSUMPTION = "energy_consumption"
    TEMPERATURE = "temperature"
    WORKLOAD_COMPLETION_TIME = "workload_completion_time"
    USER_ACTIVITY = "user_activity"

class PredictionHorizon(Enum):
    SHORT_TERM = 300      # 5 minutes
    MEDIUM_TERM = 1800    # 30 minutes
    LONG_TERM = 3600      # 1 hour
    EXTENDED_TERM = 86400 # 24 hours

@dataclass
class MetricDataPoint:
    """Single metric measurement"""
    timestamp: float
    value: float
    node_id: str
    metric_type: MetricType
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionResult:
    """Result of a prediction operation"""
    metric_type: MetricType
    node_id: str
    horizon: PredictionHorizon
    predicted_values: List[float]
    timestamps: List[float]
    confidence_intervals: List[Tuple[float, float]]
    model_accuracy: float
    prediction_time: float
    features_used: List[str]
    anomaly_detected: bool = False
    capacity_warning: bool = False
    recommendation: str = ""

@dataclass
class WorkloadPattern:
    """Identified workload pattern"""
    pattern_id: str
    pattern_type: str  # daily, weekly, burst, seasonal
    frequency: float
    amplitude: float
    phase_offset: float
    confidence: float
    first_observed: float
    last_observed: float
    occurrence_count: int

@dataclass
class CapacityPrediction:
    """Capacity planning prediction"""
    resource_type: str
    current_utilization: float
    predicted_peak: float
    time_to_capacity: Optional[float]  # seconds until capacity limit
    growth_rate: float  # per hour
    confidence: float
    recommended_action: str
    scaling_suggestions: List[str]

class ResourcePredictor:
    """Production-grade resource prediction engine with ML capabilities"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Data storage
        self.metric_history: Dict[str, List[MetricDataPoint]] = defaultdict(list)
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.workload_patterns: Dict[str, List[WorkloadPattern]] = defaultdict(list)
        
        # ML Models
        self.models: Dict[str, Dict[str, Any]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        
        # Configuration
        self.max_history_points = self.config.get('max_history_points', 10000)
        self.min_training_points = self.config.get('min_training_points', 100)
        self.prediction_cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        self.model_retrain_interval = self.config.get('retrain_interval', 3600)  # 1 hour
        
        # Performance tracking
        self.metrics = {
            'total_predictions': 0,
            'cache_hits': 0,
            'model_accuracy': {},
            'training_time': 0.0,
            'prediction_time': 0.0,
            'anomalies_detected': 0,
            'capacity_warnings': 0
        }
        
        # Last training times
        self.last_training: Dict[str, float] = {}
        
        logger.info("Resource Predictor initialized")
    
    async def record_metric(self, data_point: MetricDataPoint) -> bool:
        """Record a new metric data point"""
        try:
            key = f"{data_point.node_id}_{data_point.metric_type.value}"
            
            # Add to history
            self.metric_history[key].append(data_point)
            
            # Maintain history size limit
            if len(self.metric_history[key]) > self.max_history_points:
                self.metric_history[key] = self.metric_history[key][-self.max_history_points:]
            
            # Check for anomalies
            await self._detect_anomaly(data_point)
            
            # Invalidate related cache entries
            self._invalidate_cache(key)
            
            # Trigger retraining if needed
            await self._check_retrain_schedule(key)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
            return False
    
    async def predict_resource_usage(
        self,
        node_id: str,
        metric_type: MetricType,
        horizon: PredictionHorizon,
        force_retrain: bool = False
    ) -> PredictionResult:
        """Predict resource usage for given horizon"""
        start_time = time.time()
        
        try:
            key = f"{node_id}_{metric_type.value}_{horizon.value}"
            
            # Check cache first
            if not force_retrain and key in self.prediction_cache:
                cached = self.prediction_cache[key]
                if time.time() - cached.prediction_time < self.prediction_cache_ttl:
                    self.metrics['cache_hits'] += 1
                    return cached
            
            # Get historical data
            data_key = f"{node_id}_{metric_type.value}"
            if data_key not in self.metric_history or len(self.metric_history[data_key]) < self.min_training_points:
                return PredictionResult(
                    metric_type=metric_type,
                    node_id=node_id,
                    horizon=horizon,
                    predicted_values=[],
                    timestamps=[],
                    confidence_intervals=[],
                    model_accuracy=0.0,
                    prediction_time=time.time(),
                    features_used=[],
                    recommendation="Insufficient historical data for prediction"
                )
            
            # Train/update model if needed
            model_accuracy = await self._ensure_model_trained(data_key, force_retrain)
            
            # Generate prediction
            prediction = await self._generate_prediction(data_key, horizon)
            
            # Detect patterns
            patterns = await self._detect_patterns(data_key)
            
            # Generate recommendations
            recommendation = await self._generate_recommendation(prediction, patterns, metric_type)
            
            # Create result
            result = PredictionResult(
                metric_type=metric_type,
                node_id=node_id,
                horizon=horizon,
                predicted_values=prediction['values'],
                timestamps=prediction['timestamps'],
                confidence_intervals=prediction['confidence_intervals'],
                model_accuracy=model_accuracy,
                prediction_time=time.time(),
                features_used=prediction['features_used'],
                anomaly_detected=prediction.get('anomaly_detected', False),
                capacity_warning=prediction.get('capacity_warning', False),
                recommendation=recommendation
            )
            
            # Cache result
            self.prediction_cache[key] = result
            
            # Update metrics
            self.metrics['total_predictions'] += 1
            self.metrics['prediction_time'] = time.time() - start_time
            
            logger.info(f"Generated prediction for {node_id} {metric_type.value} (accuracy: {model_accuracy:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {node_id} {metric_type.value}: {e}")
            return PredictionResult(
                metric_type=metric_type,
                node_id=node_id,
                horizon=horizon,
                predicted_values=[],
                timestamps=[],
                confidence_intervals=[],
                model_accuracy=0.0,
                prediction_time=time.time(),
                features_used=[],
                recommendation=f"Prediction error: {str(e)}"
            )
    
    async def predict_capacity_needs(
        self,
        node_id: str,
        resource_types: List[str] = None
    ) -> List[CapacityPrediction]:
        """Predict capacity planning needs"""
        try:
            if resource_types is None:
                resource_types = ['cpu', 'memory', 'gpu', 'storage', 'network']
            
            predictions = []
            
            for resource_type in resource_types:
                # Map resource type to metric type
                metric_map = {
                    'cpu': MetricType.CPU_UTILIZATION,
                    'memory': MetricType.MEMORY_UTILIZATION,
                    'gpu': MetricType.GPU_UTILIZATION,
                    'storage': MetricType.STORAGE_IOPS,
                    'network': MetricType.NETWORK_THROUGHPUT
                }
                
                if resource_type not in metric_map:
                    continue
                
                metric_type = metric_map[resource_type]
                
                # Get long-term prediction
                prediction = await self.predict_resource_usage(
                    node_id, metric_type, PredictionHorizon.EXTENDED_TERM
                )
                
                if not prediction.predicted_values:
                    continue
                
                # Calculate capacity metrics
                current_util = await self._get_current_utilization(node_id, metric_type)
                predicted_peak = max(prediction.predicted_values) if prediction.predicted_values else current_util
                
                # Calculate time to capacity (assuming 90% is capacity limit)
                time_to_capacity = await self._calculate_time_to_capacity(
                    prediction.predicted_values, prediction.timestamps, threshold=0.9
                )
                
                # Calculate growth rate
                growth_rate = await self._calculate_growth_rate(prediction.predicted_values)
                
                # Generate recommendations
                recommended_action, scaling_suggestions = await self._generate_capacity_recommendations(
                    resource_type, current_util, predicted_peak, time_to_capacity, growth_rate
                )
                
                capacity_pred = CapacityPrediction(
                    resource_type=resource_type,
                    current_utilization=current_util,
                    predicted_peak=predicted_peak,
                    time_to_capacity=time_to_capacity,
                    growth_rate=growth_rate,
                    confidence=prediction.model_accuracy,
                    recommended_action=recommended_action,
                    scaling_suggestions=scaling_suggestions
                )
                
                predictions.append(capacity_pred)
                
                # Track warnings
                if time_to_capacity and time_to_capacity < 86400:  # Less than 24 hours
                    self.metrics['capacity_warnings'] += 1
            
            return predictions
            
        except Exception as e:
            logger.error(f"Capacity prediction failed for {node_id}: {e}")
            return []
    
    async def detect_workload_patterns(self, node_id: str) -> List[WorkloadPattern]:
        """Detect recurring workload patterns"""
        try:
            patterns = []
            
            # Analyze each metric type
            for metric_type in MetricType:
                data_key = f"{node_id}_{metric_type.value}"
                
                if data_key not in self.metric_history:
                    continue
                
                # Extract time series data
                history = self.metric_history[data_key]
                if len(history) < 100:  # Need sufficient data
                    continue
                
                timestamps = np.array([dp.timestamp for dp in history])
                values = np.array([dp.value for dp in history])
                
                # Detect patterns using FFT
                detected_patterns = await self._fft_pattern_detection(timestamps, values, metric_type.value)
                patterns.extend(detected_patterns)
                
                # Detect daily patterns
                daily_patterns = await self._detect_daily_patterns(timestamps, values, metric_type.value)
                patterns.extend(daily_patterns)
                
                # Detect burst patterns
                burst_patterns = await self._detect_burst_patterns(timestamps, values, metric_type.value)
                patterns.extend(burst_patterns)
            
            # Store patterns
            self.workload_patterns[node_id] = patterns
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed for {node_id}: {e}")
            return []
    
    async def _ensure_model_trained(self, data_key: str, force_retrain: bool = False) -> float:
        """Ensure ML model is trained and up-to-date"""
        try:
            current_time = time.time()
            
            # Check if retraining is needed
            if (force_retrain or 
                data_key not in self.models or 
                current_time - self.last_training.get(data_key, 0) > self.model_retrain_interval):
                
                logger.info(f"Training model for {data_key}")
                start_time = time.time()
                
                # Prepare training data
                X, y = await self._prepare_training_data(data_key)
                
                if len(X) < self.min_training_points:
                    return 0.0
                
                # Split data for validation
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train main prediction model
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                
                # Train anomaly detection model
                anomaly_detector = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                anomaly_detector.fit(X_train_scaled)
                
                # Calculate accuracy
                y_pred = model.predict(X_val_scaled)
                accuracy = 1.0 - mean_absolute_error(y_val, y_pred) / (np.std(y_val) + 1e-8)
                accuracy = max(0.0, min(1.0, accuracy))  # Clamp to [0, 1]
                
                # Store models
                self.models[data_key] = {
                    'model': model,
                    'accuracy': accuracy,
                    'feature_names': ['lag_1', 'lag_2', 'lag_3', 'hour', 'day_of_week', 'trend', 'seasonal']
                }
                self.scalers[data_key] = scaler
                self.anomaly_detectors[data_key] = anomaly_detector
                self.last_training[data_key] = current_time
                
                # Update metrics
                training_time = time.time() - start_time
                self.metrics['training_time'] = training_time
                self.metrics['model_accuracy'][data_key] = accuracy
                
                logger.info(f"Model trained for {data_key} (accuracy: {accuracy:.3f}, time: {training_time:.2f}s)")
                return accuracy
            
            else:
                # Return cached accuracy
                return self.models[data_key]['accuracy']
                
        except Exception as e:
            logger.error(f"Model training failed for {data_key}: {e}")
            return 0.0
    
    async def _prepare_training_data(self, data_key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with feature engineering"""
        try:
            history = self.metric_history[data_key]
            
            if len(history) < 10:
                return np.array([]), np.array([])
            
            # Extract time series
            timestamps = np.array([dp.timestamp for dp in history])
            values = np.array([dp.value for dp in history])
            
            # Create features
            features = []
            targets = []
            
            for i in range(3, len(values)):  # Need at least 3 lag values
                # Lag features
                lag_1 = values[i-1]
                lag_2 = values[i-2]
                lag_3 = values[i-3]
                
                # Time features
                dt = datetime.fromtimestamp(timestamps[i])
                hour = dt.hour / 24.0  # Normalize to [0, 1]
                day_of_week = dt.weekday() / 7.0  # Normalize to [0, 1]
                
                # Trend feature (simple linear trend over last 3 points)
                trend = (values[i-1] - values[i-3]) / 2.0
                
                # Seasonal feature (simplified)
                seasonal = math.sin(2 * math.pi * hour)  # Daily seasonality
                
                feature_vector = [lag_1, lag_2, lag_3, hour, day_of_week, trend, seasonal]
                features.append(feature_vector)
                targets.append(values[i])
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            return np.array([]), np.array([])
    
    async def _generate_prediction(self, data_key: str, horizon: PredictionHorizon) -> Dict[str, Any]:
        """Generate prediction using trained model"""
        try:
            if data_key not in self.models:
                return {
                    'values': [],
                    'timestamps': [],
                    'confidence_intervals': [],
                    'features_used': [],
                    'anomaly_detected': False,
                    'capacity_warning': False
                }
            
            model_info = self.models[data_key]
            model = model_info['model']
            scaler = self.scalers[data_key]
            anomaly_detector = self.anomaly_detectors[data_key]
            
            # Get recent history for initial conditions
            history = self.metric_history[data_key]
            recent_values = [dp.value for dp in history[-10:]]
            recent_timestamps = [dp.timestamp for dp in history[-10:]]
            
            if len(recent_values) < 3:
                return {
                    'values': [],
                    'timestamps': [],
                    'confidence_intervals': [],
                    'features_used': [],
                    'anomaly_detected': False,
                    'capacity_warning': False
                }
            
            # Generate future timestamps
            current_time = time.time()
            time_step = 60  # 1 minute intervals
            num_steps = horizon.value // time_step
            
            future_timestamps = [current_time + i * time_step for i in range(1, num_steps + 1)]
            predicted_values = []
            confidence_intervals = []
            
            # Initialize for iterative prediction
            last_values = recent_values[-3:]
            
            for i, future_time in enumerate(future_timestamps):
                # Create feature vector
                dt = datetime.fromtimestamp(future_time)
                hour = dt.hour / 24.0
                day_of_week = dt.weekday() / 7.0
                trend = (last_values[-1] - last_values[-3]) / 2.0 if len(last_values) >= 3 else 0.0
                seasonal = math.sin(2 * math.pi * hour)
                
                feature_vector = np.array([[
                    last_values[-1], last_values[-2], last_values[-3],
                    hour, day_of_week, trend, seasonal
                ]])
                
                # Scale features
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Predict
                prediction = model.predict(feature_vector_scaled)[0]
                
                # Calculate confidence interval (simplified)
                # In production, you'd use prediction intervals from the model
                base_std = np.std(recent_values[-20:]) if len(recent_values) >= 20 else 0.1
                confidence_range = base_std * 1.96  # 95% confidence
                conf_lower = max(0.0, prediction - confidence_range)
                conf_upper = min(1.0, prediction + confidence_range)
                
                predicted_values.append(prediction)
                confidence_intervals.append((conf_lower, conf_upper))
                
                # Update last_values for next iteration
                last_values = last_values[1:] + [prediction]
            
            # Detect anomalies in prediction
            if predicted_values:
                recent_scaled = scaler.transform(np.array(recent_values[-5:]).reshape(-1, 1))
                anomaly_scores = anomaly_detector.decision_function(recent_scaled)
                anomaly_detected = np.any(anomaly_scores < -0.1)  # Threshold for anomaly
            else:
                anomaly_detected = False
            
            # Check for capacity warnings
            capacity_warning = any(v > 0.85 for v in predicted_values)  # 85% threshold
            
            return {
                'values': predicted_values,
                'timestamps': future_timestamps,
                'confidence_intervals': confidence_intervals,
                'features_used': model_info['feature_names'],
                'anomaly_detected': anomaly_detected,
                'capacity_warning': capacity_warning
            }
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            return {
                'values': [],
                'timestamps': [],
                'confidence_intervals': [],
                'features_used': [],
                'anomaly_detected': False,
                'capacity_warning': False
            }
    
    async def _detect_anomaly(self, data_point: MetricDataPoint):
        """Detect if a data point is anomalous"""
        try:
            data_key = f"{data_point.node_id}_{data_point.metric_type.value}"
            
            if data_key not in self.anomaly_detectors:
                return False
            
            detector = self.anomaly_detectors[data_key]
            scaler = self.scalers.get(data_key)
            
            if not scaler:
                return False
            
            # Prepare data point for anomaly detection
            feature_vector = np.array([[data_point.value]])
            feature_scaled = scaler.transform(feature_vector)
            
            # Check for anomaly
            anomaly_score = detector.decision_function(feature_scaled)[0]
            is_anomaly = anomaly_score < -0.1
            
            if is_anomaly:
                self.metrics['anomalies_detected'] += 1
                logger.warning(f"Anomaly detected in {data_key}: value={data_point.value}, score={anomaly_score}")
            
            return is_anomaly
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False
    
    async def _detect_patterns(self, data_key: str) -> List[WorkloadPattern]:
        """Detect patterns in historical data"""
        # This is a simplified implementation
        # In production, you'd use more sophisticated pattern detection
        return []
    
    async def _fft_pattern_detection(self, timestamps: np.ndarray, values: np.ndarray, metric_name: str) -> List[WorkloadPattern]:
        """Use FFT to detect periodic patterns"""
        try:
            if len(values) < 100:
                return []
            
            # Perform FFT
            fft = np.fft.fft(values)
            freqs = np.fft.fftfreq(len(values))
            
            # Find dominant frequencies
            power = np.abs(fft)
            dominant_indices = np.argsort(power)[-5:]  # Top 5 frequencies
            
            patterns = []
            for idx in dominant_indices:
                if freqs[idx] > 0:  # Only positive frequencies
                    frequency = freqs[idx]
                    amplitude = power[idx] / len(values)  # Normalize
                    
                    if amplitude > 0.1:  # Threshold for significant patterns
                        pattern = WorkloadPattern(
                            pattern_id=f"{metric_name}_fft_{idx}",
                            pattern_type="periodic",
                            frequency=frequency,
                            amplitude=amplitude,
                            phase_offset=np.angle(fft[idx]),
                            confidence=min(amplitude * 2, 1.0),
                            first_observed=timestamps[0],
                            last_observed=timestamps[-1],
                            occurrence_count=int(len(values) * frequency)
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"FFT pattern detection failed: {e}")
            return []
    
    async def _detect_daily_patterns(self, timestamps: np.ndarray, values: np.ndarray, metric_name: str) -> List[WorkloadPattern]:
        """Detect daily recurring patterns"""
        try:
            if len(values) < 1440:  # Need at least 24 hours of minute data
                return []
            
            # Group by hour of day
            hourly_stats = defaultdict(list)
            for ts, val in zip(timestamps, values):
                hour = datetime.fromtimestamp(ts).hour
                hourly_stats[hour].append(val)
            
            # Calculate hourly averages and variance
            hourly_avg = {}
            hourly_var = {}
            for hour, vals in hourly_stats.items():
                if len(vals) > 1:
                    hourly_avg[hour] = np.mean(vals)
                    hourly_var[hour] = np.var(vals)
            
            if len(hourly_avg) < 12:  # Need data for at least half the day
                return []
            
            # Detect if there's a significant daily pattern
            daily_values = [hourly_avg.get(h, 0) for h in range(24)]
            daily_variance = np.var(daily_values)
            overall_variance = np.var(values)
            
            if daily_variance > overall_variance * 0.5:  # Significant daily pattern
                pattern = WorkloadPattern(
                    pattern_id=f"{metric_name}_daily",
                    pattern_type="daily",
                    frequency=1.0 / 86400,  # Once per day
                    amplitude=np.sqrt(daily_variance),
                    phase_offset=0.0,
                    confidence=min(daily_variance / overall_variance, 1.0),
                    first_observed=timestamps[0],
                    last_observed=timestamps[-1],
                    occurrence_count=len(timestamps) // 1440
                )
                return [pattern]
            
            return []
            
        except Exception as e:
            logger.error(f"Daily pattern detection failed: {e}")
            return []
    
    async def _detect_burst_patterns(self, timestamps: np.ndarray, values: np.ndarray, metric_name: str) -> List[WorkloadPattern]:
        """Detect burst/spike patterns"""
        try:
            if len(values) < 50:
                return []
            
            # Calculate moving average and standard deviation
            window_size = min(20, len(values) // 5)
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            moving_std = np.array([np.std(values[max(0, i-window_size):i+1]) for i in range(len(values))])
            
            # Detect spikes (values significantly above moving average)
            spike_threshold = 2.0  # 2 standard deviations
            spikes = []
            
            for i in range(window_size-1, len(values)):
                if i-window_size+1 < len(moving_avg):
                    expected = moving_avg[i-window_size+1]
                    std_dev = moving_std[i]
                    
                    if values[i] > expected + spike_threshold * std_dev:
                        spikes.append((timestamps[i], values[i]))
            
            if len(spikes) > 5:  # Need multiple spikes to consider it a pattern
                # Calculate burst frequency
                spike_times = [s[0] for s in spikes]
                time_diffs = np.diff(spike_times)
                avg_interval = np.mean(time_diffs) if len(time_diffs) > 0 else 0
                
                if avg_interval > 0:
                    pattern = WorkloadPattern(
                        pattern_id=f"{metric_name}_burst",
                        pattern_type="burst",
                        frequency=1.0 / avg_interval,
                        amplitude=np.mean([s[1] for s in spikes]),
                        phase_offset=0.0,
                        confidence=min(len(spikes) / 50.0, 1.0),
                        first_observed=timestamps[0],
                        last_observed=timestamps[-1],
                        occurrence_count=len(spikes)
                    )
                    return [pattern]
            
            return []
            
        except Exception as e:
            logger.error(f"Burst pattern detection failed: {e}")
            return []
    
    async def _generate_recommendation(self, prediction: Dict[str, Any], patterns: List[WorkloadPattern], metric_type: MetricType) -> str:
        """Generate actionable recommendations based on predictions"""
        try:
            if not prediction['values']:
                return "Insufficient data for recommendations"
            
            max_predicted = max(prediction['values'])
            avg_predicted = np.mean(prediction['values'])
            
            recommendations = []
            
            # Capacity recommendations
            if max_predicted > 0.9:
                recommendations.append("Critical: Resource utilization will exceed 90%")
            elif max_predicted > 0.8:
                recommendations.append("Warning: High resource utilization predicted")
            
            # Pattern-based recommendations
            daily_patterns = [p for p in patterns if p.pattern_type == "daily"]
            if daily_patterns and daily_patterns[0].confidence > 0.7:
                recommendations.append("Strong daily pattern detected - consider scheduled scaling")
            
            burst_patterns = [p for p in patterns if p.pattern_type == "burst"]
            if burst_patterns and burst_patterns[0].confidence > 0.6:
                recommendations.append("Burst pattern detected - enable auto-scaling for spikes")
            
            # Anomaly recommendations
            if prediction.get('anomaly_detected'):
                recommendations.append("Anomaly detected - investigate potential issues")
            
            # Metric-specific recommendations
            if metric_type == MetricType.CPU_UTILIZATION:
                if avg_predicted > 0.7:
                    recommendations.append("Consider CPU scaling or workload redistribution")
            elif metric_type == MetricType.MEMORY_UTILIZATION:
                if avg_predicted > 0.8:
                    recommendations.append("Memory pressure expected - increase memory allocation")
            elif metric_type == MetricType.ENERGY_CONSUMPTION:
                if avg_predicted > 0.8:
                    recommendations.append("High energy consumption predicted - optimize workloads")
            
            return " | ".join(recommendations) if recommendations else "No specific recommendations"
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return "Unable to generate recommendations"
    
    async def _get_current_utilization(self, node_id: str, metric_type: MetricType) -> float:
        """Get current utilization for a resource"""
        try:
            data_key = f"{node_id}_{metric_type.value}"
            if data_key in self.metric_history and self.metric_history[data_key]:
                return self.metric_history[data_key][-1].value
            return 0.0
        except:
            return 0.0
    
    async def _calculate_time_to_capacity(self, predicted_values: List[float], timestamps: List[float], threshold: float = 0.9) -> Optional[float]:
        """Calculate time until capacity threshold is reached"""
        try:
            current_time = time.time()
            
            for i, (value, timestamp) in enumerate(zip(predicted_values, timestamps)):
                if value >= threshold:
                    return timestamp - current_time
            
            return None  # Threshold not reached in prediction horizon
            
        except:
            return None
    
    async def _calculate_growth_rate(self, predicted_values: List[float]) -> float:
        """Calculate resource growth rate per hour"""
        try:
            if len(predicted_values) < 2:
                return 0.0
            
            # Simple linear regression to find growth rate
            x = np.arange(len(predicted_values))
            y = np.array(predicted_values)
            
            slope, _ = np.polyfit(x, y, 1)
            
            # Convert to per-hour rate (assuming 1-minute intervals)
            growth_rate_per_hour = slope * 60
            
            return growth_rate_per_hour
            
        except:
            return 0.0
    
    async def _generate_capacity_recommendations(
        self,
        resource_type: str,
        current_util: float,
        predicted_peak: float,
        time_to_capacity: Optional[float],
        growth_rate: float
    ) -> Tuple[str, List[str]]:
        """Generate capacity planning recommendations"""
        try:
            action = "monitor"
            suggestions = []
            
            if predicted_peak > 0.9:
                action = "scale_immediately"
                suggestions.append(f"Scale {resource_type} capacity immediately")
                suggestions.append("Consider workload redistribution")
            elif predicted_peak > 0.8:
                action = "prepare_scaling"
                suggestions.append(f"Prepare to scale {resource_type} capacity")
                suggestions.append("Monitor closely for next 24 hours")
            elif predicted_peak > 0.7:
                action = "monitor_closely"
                suggestions.append(f"Monitor {resource_type} usage trends")
            
            if time_to_capacity and time_to_capacity < 3600:  # Less than 1 hour
                action = "scale_immediately"
                suggestions.append("URGENT: Capacity limit in less than 1 hour")
            elif time_to_capacity and time_to_capacity < 86400:  # Less than 24 hours
                suggestions.append(f"Capacity limit in {time_to_capacity/3600:.1f} hours")
            
            if growth_rate > 0.1:  # 10% per hour
                suggestions.append("High growth rate detected - enable auto-scaling")
            
            return action, suggestions
            
        except Exception as e:
            logger.error(f"Capacity recommendation generation failed: {e}")
            return "monitor", ["Unable to generate recommendations"]
    
    def _invalidate_cache(self, data_key: str):
        """Invalidate prediction cache entries related to data key"""
        try:
            keys_to_remove = [k for k in self.prediction_cache.keys() if data_key in k]
            for key in keys_to_remove:
                del self.prediction_cache[key]
        except:
            pass
    
    async def _check_retrain_schedule(self, data_key: str):
        """Check if model retraining is needed"""
        try:
            current_time = time.time()
            last_train = self.last_training.get(data_key, 0)
            
            if current_time - last_train > self.model_retrain_interval:
                # Schedule retraining (in production, this might be async)
                await self._ensure_model_trained(data_key, force_retrain=True)
        except Exception as e:
            logger.error(f"Retrain schedule check failed: {e}")
    
    async def get_prediction_metrics(self) -> Dict[str, Any]:
        """Get comprehensive prediction engine metrics"""
        try:
            return {
                'performance': {
                    'total_predictions': self.metrics['total_predictions'],
                    'cache_hit_rate': self.metrics['cache_hits'] / max(self.metrics['total_predictions'], 1),
                    'average_prediction_time': self.metrics['prediction_time'],
                    'average_training_time': self.metrics['training_time']
                },
                'model_quality': {
                    'model_accuracies': dict(self.metrics['model_accuracy']),
                    'average_accuracy': np.mean(list(self.metrics['model_accuracy'].values())) if self.metrics['model_accuracy'] else 0.0,
                    'trained_models': len(self.models)
                },
                'anomaly_detection': {
                    'anomalies_detected': self.metrics['anomalies_detected'],
                    'anomaly_rate': self.metrics['anomalies_detected'] / max(sum(len(h) for h in self.metric_history.values()), 1)
                },
                'capacity_planning': {
                    'capacity_warnings': self.metrics['capacity_warnings'],
                    'workload_patterns_detected': sum(len(patterns) for patterns in self.workload_patterns.values())
                },
                'data_status': {
                    'total_data_points': sum(len(h) for h in self.metric_history.values()),
                    'cache_entries': len(self.prediction_cache),
                    'active_data_streams': len(self.metric_history)
                }
            }
        except Exception as e:
            logger.error(f"Error generating prediction metrics: {e}")
            return {}

# Global predictor instance
_predictor_instance = None

def get_predictor() -> ResourcePredictor:
    """Get or create global predictor instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ResourcePredictor()
    return _predictor_instance

async def initialize_predictor(config: Dict[str, Any] = None):
    """Initialize the global predictor"""
    global _predictor_instance
    _predictor_instance = ResourcePredictor(config)
    logger.info("Global resource predictor initialized")
    return _predictor_instance
