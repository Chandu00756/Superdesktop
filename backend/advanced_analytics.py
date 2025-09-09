"""
Omega Super Desktop Console v2.0 - Advanced Analytics Engine
Enterprise-grade analytics with ML models, real-time dashboards, and predictive insights
"""

import asyncio
import json
import logging
import time
import uuid
import math
import statistics
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class AggregationType(Enum):
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"
    STDDEV = "stddev"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class TimeWindow(Enum):
    MINUTE = 60
    HOUR = 3600
    DAY = 86400
    WEEK = 604800
    MONTH = 2592000

@dataclass
class Metric:
    """Individual metric data point"""
    metric_id: str
    name: str
    value: float
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    unit: Optional[str] = None

@dataclass
class Alert:
    """Analytics alert definition"""
    alert_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "> 80", "< 10", "= 0"
    threshold: float
    severity: AlertSeverity
    time_window: TimeWindow
    enabled: bool = True
    actions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_triggered: Optional[float] = None
    trigger_count: int = 0

@dataclass
class Dashboard:
    """Analytics dashboard configuration"""
    dashboard_id: str
    name: str
    description: str
    widgets: List[Dict[str, Any]] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    auto_refresh: int = 30  # seconds
    created_at: float = field(default_factory=time.time)
    owner: Optional[str] = None
    shared: bool = False

@dataclass
class MLModel:
    """Machine learning model for analytics"""
    model_id: str
    name: str
    model_type: str  # linear_regression, random_forest, neural_network, etc.
    target_metric: str
    features: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    training_data_size: int = 0
    accuracy: float = 0.0
    last_trained: Optional[float] = None
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True

class MetricAggregator:
    """Aggregate metrics over time windows"""
    
    def __init__(self):
        self.raw_metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = {}
        
    async def add_metric(self, metric: Metric):
        """Add metric to aggregation pool"""
        try:
            self.raw_metrics[metric.name].append(metric)
            
            # Keep only recent metrics (last 24 hours)
            cutoff_time = time.time() - 86400
            self.raw_metrics[metric.name] = [
                m for m in self.raw_metrics[metric.name] 
                if m.timestamp > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Failed to add metric: {e}")
            
    async def aggregate_metrics(self, metric_name: str, time_window: TimeWindow, 
                              aggregation: AggregationType) -> Optional[float]:
        """Aggregate metrics over time window"""
        try:
            if metric_name not in self.raw_metrics:
                return None
                
            cutoff_time = time.time() - time_window.value
            recent_metrics = [
                m for m in self.raw_metrics[metric_name]
                if m.timestamp > cutoff_time
            ]
            
            if not recent_metrics:
                return None
                
            values = [m.value for m in recent_metrics]
            
            if aggregation == AggregationType.SUM:
                return sum(values)
            elif aggregation == AggregationType.AVG:
                return statistics.mean(values)
            elif aggregation == AggregationType.MIN:
                return min(values)
            elif aggregation == AggregationType.MAX:
                return max(values)
            elif aggregation == AggregationType.COUNT:
                return len(values)
            elif aggregation == AggregationType.STDDEV:
                return statistics.stdev(values) if len(values) > 1 else 0
            elif aggregation == AggregationType.PERCENTILE:
                # Default to 95th percentile
                return self._percentile(values, 95)
            else:
                return statistics.mean(values)
                
        except Exception as e:
            logger.error(f"Metric aggregation failed: {e}")
            return None
            
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value"""
        try:
            sorted_values = sorted(values)
            k = (len(sorted_values) - 1) * (percentile / 100)
            f = math.floor(k)
            c = math.ceil(k)
            
            if f == c:
                return sorted_values[int(k)]
            else:
                return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)
                
        except Exception:
            return 0.0
            
    async def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get comprehensive metric summary"""
        try:
            if metric_name not in self.raw_metrics:
                return {}
                
            recent_metrics = self.raw_metrics[metric_name][-1000:]  # Last 1000 points
            values = [m.value for m in recent_metrics]
            
            if not values:
                return {}
                
            summary = {
                'count': len(values),
                'sum': sum(values),
                'avg': statistics.mean(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1] if values else 0,
                'first_timestamp': recent_metrics[0].timestamp if recent_metrics else 0,
                'last_timestamp': recent_metrics[-1].timestamp if recent_metrics else 0
            }
            
            if len(values) > 1:
                summary['stddev'] = statistics.stdev(values)
                summary['p50'] = self._percentile(values, 50)
                summary['p95'] = self._percentile(values, 95)
                summary['p99'] = self._percentile(values, 99)
            else:
                summary['stddev'] = 0
                summary['p50'] = values[0] if values else 0
                summary['p95'] = values[0] if values else 0
                summary['p99'] = values[0] if values else 0
                
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get metric summary: {e}")
            return {}

class AlertManager:
    """Manage analytics alerts and notifications"""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
    async def add_alert(self, alert: Alert) -> bool:
        """Add new alert definition"""
        try:
            self.alerts[alert.alert_id] = alert
            logger.info(f"Added alert: {alert.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add alert: {e}")
            return False
            
    async def evaluate_alerts(self, aggregator: MetricAggregator) -> List[Dict[str, Any]]:
        """Evaluate all alerts against current metrics"""
        triggered_alerts = []
        
        try:
            for alert in self.alerts.values():
                if not alert.enabled:
                    continue
                    
                # Get aggregated metric value
                metric_value = await aggregator.aggregate_metrics(
                    alert.metric_name,
                    alert.time_window,
                    AggregationType.AVG
                )
                
                if metric_value is None:
                    continue
                    
                # Evaluate condition
                triggered = self._evaluate_condition(metric_value, alert.condition, alert.threshold)
                
                if triggered:
                    alert_event = {
                        'alert_id': alert.alert_id,
                        'alert_name': alert.name,
                        'metric_name': alert.metric_name,
                        'current_value': metric_value,
                        'threshold': alert.threshold,
                        'condition': alert.condition,
                        'severity': alert.severity.value,
                        'timestamp': time.time(),
                        'actions': alert.actions
                    }
                    
                    triggered_alerts.append(alert_event)
                    self.alert_history.append(alert_event)
                    
                    # Update alert statistics
                    alert.last_triggered = time.time()
                    alert.trigger_count += 1
                    
                    logger.warning(f"Alert triggered: {alert.name} - {metric_value} {alert.condition} {alert.threshold}")
                    
            # Keep only recent alert history
            cutoff_time = time.time() - 86400  # 24 hours
            self.alert_history = [
                alert for alert in self.alert_history
                if alert['timestamp'] > cutoff_time
            ]
            
            return triggered_alerts
            
        except Exception as e:
            logger.error(f"Alert evaluation failed: {e}")
            return []
            
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        try:
            condition = condition.strip()
            
            if condition.startswith('>='):
                return value >= threshold
            elif condition.startswith('<='):
                return value <= threshold
            elif condition.startswith('>'):
                return value > threshold
            elif condition.startswith('<'):
                return value < threshold
            elif condition.startswith('==') or condition.startswith('='):
                return abs(value - threshold) < 0.001  # Float equality
            elif condition.startswith('!='):
                return abs(value - threshold) >= 0.001
            else:
                return value > threshold  # Default to greater than
                
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
            
    async def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert system summary"""
        try:
            total_alerts = len(self.alerts)
            enabled_alerts = len([a for a in self.alerts.values() if a.enabled])
            recent_triggers = len([
                a for a in self.alert_history
                if a['timestamp'] > time.time() - 3600  # Last hour
            ])
            
            severity_counts = defaultdict(int)
            for alert in self.alert_history:
                if alert['timestamp'] > time.time() - 3600:
                    severity_counts[alert['severity']] += 1
                    
            return {
                'total_alerts': total_alerts,
                'enabled_alerts': enabled_alerts,
                'recent_triggers': recent_triggers,
                'severity_breakdown': dict(severity_counts),
                'alert_history_size': len(self.alert_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get alert summary: {e}")
            return {}

class MLPredictor:
    """Machine learning predictor for analytics"""
    
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        
    async def create_model(self, model: MLModel) -> bool:
        """Create new ML model"""
        try:
            # Initialize model with basic linear regression
            model.parameters = {
                'coefficients': [],
                'intercept': 0.0,
                'feature_means': [],
                'feature_stds': []
            }
            
            self.models[model.model_id] = model
            logger.info(f"Created ML model: {model.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return False
            
    async def train_model(self, model_id: str, aggregator: MetricAggregator) -> bool:
        """Train ML model with historical data"""
        try:
            if model_id not in self.models:
                return False
                
            model = self.models[model_id]
            
            # Get training data
            training_data = await self._prepare_training_data(model, aggregator)
            
            if len(training_data['X']) < 10:  # Need minimum data points
                logger.warning(f"Insufficient training data for model {model.name}")
                return False
                
            # Simple linear regression implementation
            X = training_data['X']
            y = training_data['y']
            
            # Calculate coefficients using least squares method
            coefficients, intercept = self._linear_regression(X, y)
            
            # Update model parameters
            model.parameters['coefficients'] = coefficients
            model.parameters['intercept'] = intercept
            model.parameters['feature_means'] = [statistics.mean(feature) for feature in zip(*X)]
            model.parameters['feature_stds'] = [statistics.stdev(feature) if len(feature) > 1 else 1.0 for feature in zip(*X)]
            
            model.training_data_size = len(X)
            model.last_trained = time.time()
            
            # Calculate accuracy on training data
            predictions = [self._predict_single(x, model) for x in X]
            model.accuracy = self._calculate_accuracy(y, predictions)
            
            logger.info(f"Trained model {model.name} with {len(X)} samples, accuracy: {model.accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
            
    async def predict(self, model_id: str, features: List[float]) -> Optional[float]:
        """Make prediction using trained model"""
        try:
            if model_id not in self.models:
                return None
                
            model = self.models[model_id]
            
            if not model.parameters.get('coefficients'):
                logger.warning(f"Model {model.name} not trained")
                return None
                
            return self._predict_single(features, model)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
            
    async def predict_trend(self, model_id: str, aggregator: MetricAggregator, 
                          future_points: int = 10) -> List[Dict[str, Any]]:
        """Predict future trend for metric"""
        try:
            if model_id not in self.models:
                return []
                
            model = self.models[model_id]
            predictions = []
            
            # Get recent data for trend prediction
            recent_data = await self._prepare_prediction_data(model, aggregator)
            
            if not recent_data:
                return []
                
            # Generate future predictions
            current_time = time.time()
            
            for i in range(future_points):
                future_time = current_time + (i + 1) * 300  # 5-minute intervals
                
                # Create feature vector for future time
                features = self._create_feature_vector(future_time, recent_data)
                
                predicted_value = await self.predict(model_id, features)
                
                if predicted_value is not None:
                    predictions.append({
                        'timestamp': future_time,
                        'predicted_value': predicted_value,
                        'confidence': max(0.1, model.accuracy)  # Use model accuracy as confidence
                    })
                    
            return predictions
            
        except Exception as e:
            logger.error(f"Trend prediction failed: {e}")
            return []
            
    async def _prepare_training_data(self, model: MLModel, aggregator: MetricAggregator) -> Dict[str, List]:
        """Prepare training data for model"""
        try:
            X = []  # Features
            y = []  # Target values
            
            # Get historical data for target metric
            if model.target_metric not in aggregator.raw_metrics:
                return {'X': [], 'y': []}
                
            target_metrics = aggregator.raw_metrics[model.target_metric]
            
            # Use sliding window to create training samples
            window_size = 10
            
            for i in range(window_size, len(target_metrics)):
                # Create feature vector from previous values
                features = []
                
                # Add temporal features
                current_time = target_metrics[i].timestamp
                features.extend([
                    current_time % 86400,  # Time of day
                    (current_time % 604800) / 86400,  # Day of week
                    math.sin(2 * math.pi * (current_time % 86400) / 86400),  # Daily cycle
                    math.cos(2 * math.pi * (current_time % 86400) / 86400)   # Daily cycle
                ])
                
                # Add previous values as features
                for j in range(window_size):
                    features.append(target_metrics[i - window_size + j].value)
                    
                X.append(features)
                y.append(target_metrics[i].value)
                
            return {'X': X, 'y': y}
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            return {'X': [], 'y': []}
            
    async def _prepare_prediction_data(self, model: MLModel, aggregator: MetricAggregator) -> List[Dict[str, Any]]:
        """Prepare recent data for prediction"""
        try:
            if model.target_metric not in aggregator.raw_metrics:
                return []
                
            recent_metrics = aggregator.raw_metrics[model.target_metric][-100:]  # Last 100 points
            
            return [
                {
                    'timestamp': m.timestamp,
                    'value': m.value
                }
                for m in recent_metrics
            ]
            
        except Exception as e:
            logger.error(f"Prediction data preparation failed: {e}")
            return []
            
    def _linear_regression(self, X: List[List[float]], y: List[float]) -> Tuple[List[float], float]:
        """Simple linear regression implementation"""
        try:
            n = len(X)
            m = len(X[0]) if X else 0
            
            if n == 0 or m == 0:
                return [0.0] * m, 0.0
                
            # Convert to matrices (simplified implementation)
            X_means = [sum(X[i][j] for i in range(n)) / n for j in range(m)]
            y_mean = sum(y) / n
            
            # Calculate coefficients using normal equation approximation
            coefficients = []
            
            for j in range(m):
                # Simple correlation-based coefficient
                numerator = sum((X[i][j] - X_means[j]) * (y[i] - y_mean) for i in range(n))
                denominator = sum((X[i][j] - X_means[j]) ** 2 for i in range(n))
                
                if denominator > 0:
                    coefficients.append(numerator / denominator)
                else:
                    coefficients.append(0.0)
                    
            # Calculate intercept
            intercept = y_mean - sum(coefficients[j] * X_means[j] for j in range(m))
            
            return coefficients, intercept
            
        except Exception as e:
            logger.error(f"Linear regression failed: {e}")
            return [0.0] * len(X[0]) if X else [], 0.0
            
    def _predict_single(self, features: List[float], model: MLModel) -> float:
        """Make single prediction"""
        try:
            coefficients = model.parameters.get('coefficients', [])
            intercept = model.parameters.get('intercept', 0.0)
            
            if len(coefficients) != len(features):
                return 0.0
                
            prediction = intercept + sum(coef * feat for coef, feat in zip(coefficients, features))
            return prediction
            
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            return 0.0
            
    def _calculate_accuracy(self, actual: List[float], predicted: List[float]) -> float:
        """Calculate prediction accuracy"""
        try:
            if len(actual) != len(predicted) or len(actual) == 0:
                return 0.0
                
            # Calculate R-squared
            actual_mean = statistics.mean(actual)
            ss_res = sum((actual[i] - predicted[i]) ** 2 for i in range(len(actual)))
            ss_tot = sum((actual[i] - actual_mean) ** 2 for i in range(len(actual)))
            
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
                
            r_squared = 1 - (ss_res / ss_tot)
            return max(0.0, min(1.0, r_squared))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {e}")
            return 0.0
            
    def _create_feature_vector(self, timestamp: float, recent_data: List[Dict[str, Any]]) -> List[float]:
        """Create feature vector for prediction"""
        try:
            features = []
            
            # Add temporal features
            features.extend([
                timestamp % 86400,  # Time of day
                (timestamp % 604800) / 86400,  # Day of week
                math.sin(2 * math.pi * (timestamp % 86400) / 86400),  # Daily cycle
                math.cos(2 * math.pi * (timestamp % 86400) / 86400)   # Daily cycle
            ])
            
            # Add recent values as features
            recent_values = [data['value'] for data in recent_data[-10:]]  # Last 10 values
            
            # Pad with zeros if not enough data
            while len(recent_values) < 10:
                recent_values.insert(0, 0.0)
                
            features.extend(recent_values)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature vector creation failed: {e}")
            return [0.0] * 14  # Default feature vector size

class DashboardManager:
    """Manage analytics dashboards"""
    
    def __init__(self):
        self.dashboards: Dict[str, Dashboard] = {}
        
    async def create_dashboard(self, dashboard: Dashboard) -> bool:
        """Create new dashboard"""
        try:
            self.dashboards[dashboard.dashboard_id] = dashboard
            logger.info(f"Created dashboard: {dashboard.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return False
            
    async def render_dashboard_data(self, dashboard_id: str, aggregator: MetricAggregator) -> Dict[str, Any]:
        """Render dashboard with current data"""
        try:
            if dashboard_id not in self.dashboards:
                return {}
                
            dashboard = self.dashboards[dashboard_id]
            rendered_data = {
                'dashboard_id': dashboard_id,
                'name': dashboard.name,
                'description': dashboard.description,
                'last_updated': time.time(),
                'widgets': []
            }
            
            # Render each widget
            for widget_config in dashboard.widgets:
                widget_data = await self._render_widget(widget_config, aggregator)
                rendered_data['widgets'].append(widget_data)
                
            return rendered_data
            
        except Exception as e:
            logger.error(f"Dashboard rendering failed: {e}")
            return {}
            
    async def _render_widget(self, widget_config: Dict[str, Any], aggregator: MetricAggregator) -> Dict[str, Any]:
        """Render individual widget"""
        try:
            widget_type = widget_config.get('type', 'metric')
            metric_name = widget_config.get('metric', '')
            
            widget_data = {
                'id': widget_config.get('id', str(uuid.uuid4())),
                'type': widget_type,
                'title': widget_config.get('title', metric_name),
                'data': None,
                'timestamp': time.time()
            }
            
            if widget_type == 'metric':
                # Single metric display
                summary = await aggregator.get_metric_summary(metric_name)
                widget_data['data'] = summary
                
            elif widget_type == 'chart':
                # Time series chart
                if metric_name in aggregator.raw_metrics:
                    recent_metrics = aggregator.raw_metrics[metric_name][-100:]  # Last 100 points
                    widget_data['data'] = {
                        'series': [
                            {
                                'timestamp': m.timestamp,
                                'value': m.value
                            }
                            for m in recent_metrics
                        ]
                    }
                    
            elif widget_type == 'gauge':
                # Gauge display
                current_value = await aggregator.aggregate_metrics(
                    metric_name, TimeWindow.MINUTE, AggregationType.AVG
                )
                
                widget_data['data'] = {
                    'current_value': current_value or 0,
                    'min_value': widget_config.get('min_value', 0),
                    'max_value': widget_config.get('max_value', 100),
                    'threshold': widget_config.get('threshold', 80)
                }
                
            elif widget_type == 'table':
                # Data table
                metrics = widget_config.get('metrics', [])
                table_data = []
                
                for metric in metrics:
                    summary = await aggregator.get_metric_summary(metric)
                    if summary:
                        table_data.append({
                            'metric': metric,
                            'current': summary.get('latest', 0),
                            'avg': summary.get('avg', 0),
                            'min': summary.get('min', 0),
                            'max': summary.get('max', 0)
                        })
                        
                widget_data['data'] = {'rows': table_data}
                
            return widget_data
            
        except Exception as e:
            logger.error(f"Widget rendering failed: {e}")
            return {'id': str(uuid.uuid4()), 'type': 'error', 'data': None}

class AdvancedAnalyticsEngine:
    """Main advanced analytics engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.db_path = self.config.get('db_path', 'backend/analytics.db')
        
        # Core components
        self.aggregator = MetricAggregator()
        self.alert_manager = AlertManager()
        self.ml_predictor = MLPredictor()
        self.dashboard_manager = DashboardManager()
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Monitoring
        self.running = False
        
        # Metrics
        self.metrics = {
            'total_metrics': 0,
            'active_alerts': 0,
            'trained_models': 0,
            'dashboard_count': 0,
            'prediction_accuracy': 0.0
        }
        
    async def initialize(self) -> bool:
        """Initialize the analytics engine"""
        try:
            logger.info("Initializing Advanced Analytics Engine...")
            
            # Setup database
            await self._setup_database()
            
            # Load configurations
            await self._load_configurations()
            
            # Create default dashboards
            await self._create_default_dashboards()
            
            # Start background tasks
            self.running = True
            asyncio.create_task(self._analytics_processor())
            asyncio.create_task(self._alert_monitor())
            asyncio.create_task(self._model_trainer())
            
            logger.info("Advanced Analytics Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Analytics engine initialization failed: {e}")
            return False
            
    async def shutdown(self):
        """Shutdown the analytics engine"""
        try:
            logger.info("Shutting down Advanced Analytics Engine...")
            self.running = False
            
            logger.info("Advanced Analytics Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Analytics engine shutdown error: {e}")
            
    async def add_metric(self, metric: Metric) -> bool:
        """Add metric to analytics system"""
        try:
            await self.aggregator.add_metric(metric)
            self.metrics['total_metrics'] += 1
            
            await self._emit_event('metric_added', {
                'metric_name': metric.name,
                'value': metric.value,
                'timestamp': metric.timestamp
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add metric: {e}")
            return False
            
    async def create_alert(self, alert: Alert) -> bool:
        """Create new analytics alert"""
        try:
            success = await self.alert_manager.add_alert(alert)
            if success:
                self.metrics['active_alerts'] += 1
                
                await self._emit_event('alert_created', {
                    'alert_id': alert.alert_id,
                    'alert_name': alert.name,
                    'metric_name': alert.metric_name
                })
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return False
            
    async def create_ml_model(self, model: MLModel) -> bool:
        """Create machine learning model"""
        try:
            success = await self.ml_predictor.create_model(model)
            if success:
                self.metrics['trained_models'] += 1
                
                await self._emit_event('model_created', {
                    'model_id': model.model_id,
                    'model_name': model.name,
                    'target_metric': model.target_metric
                })
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return False
            
    async def train_model(self, model_id: str) -> bool:
        """Train machine learning model"""
        try:
            success = await self.ml_predictor.train_model(model_id, self.aggregator)
            
            if success:
                await self._emit_event('model_trained', {
                    'model_id': model_id,
                    'training_time': time.time()
                })
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            return False
            
    async def predict_metric(self, model_id: str, features: List[float]) -> Optional[float]:
        """Make prediction using ML model"""
        try:
            return await self.ml_predictor.predict(model_id, features)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
            
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        try:
            # Aggregate system metrics
            metric_summaries = {}
            for metric_name in self.aggregator.raw_metrics.keys():
                summary = await self.aggregator.get_metric_summary(metric_name)
                metric_summaries[metric_name] = summary
                
            # Alert summary
            alert_summary = await self.alert_manager.get_alert_summary()
            
            # Model performance
            model_performance = {}
            for model_id, model in self.ml_predictor.models.items():
                model_performance[model_id] = {
                    'name': model.name,
                    'accuracy': model.accuracy,
                    'training_data_size': model.training_data_size,
                    'last_trained': model.last_trained
                }
                
            return {
                'analytics_status': {
                    'running': self.running,
                    'total_metrics': len(self.aggregator.raw_metrics),
                    'total_alerts': len(self.alert_manager.alerts),
                    'total_models': len(self.ml_predictor.models),
                    'total_dashboards': len(self.dashboard_manager.dashboards)
                },
                'metric_summaries': metric_summaries,
                'alert_summary': alert_summary,
                'model_performance': model_performance,
                'system_metrics': self.metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}
            
    async def _analytics_processor(self):
        """Background analytics processing"""
        while self.running:
            try:
                # Process metrics and update aggregations
                current_time = time.time()
                
                # Clean up old metrics
                for metric_list in self.aggregator.raw_metrics.values():
                    cutoff_time = current_time - 86400  # 24 hours
                    metric_list[:] = [m for m in metric_list if m.timestamp > cutoff_time]
                    
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                logger.error(f"Analytics processing error: {e}")
                await asyncio.sleep(60)
                
    async def _alert_monitor(self):
        """Monitor alerts and trigger notifications"""
        while self.running:
            try:
                triggered_alerts = await self.alert_manager.evaluate_alerts(self.aggregator)
                
                for alert_event in triggered_alerts:
                    await self._emit_event('alert_triggered', alert_event)
                    
                    # Execute alert actions
                    for action in alert_event.get('actions', []):
                        await self._execute_alert_action(action, alert_event)
                        
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Alert monitoring error: {e}")
                await asyncio.sleep(30)
                
    async def _model_trainer(self):
        """Background model training and retraining"""
        while self.running:
            try:
                current_time = time.time()
                
                # Retrain models that need updating
                for model_id, model in self.ml_predictor.models.items():
                    if not model.enabled:
                        continue
                        
                    # Retrain if no training or if 24 hours since last training
                    if (model.last_trained is None or 
                        current_time - model.last_trained > 86400):
                        
                        logger.info(f"Retraining model: {model.name}")
                        await self.ml_predictor.train_model(model_id, self.aggregator)
                        
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Model training error: {e}")
                await asyncio.sleep(3600)
                
    async def _execute_alert_action(self, action: str, alert_event: Dict[str, Any]):
        """Execute alert action"""
        try:
            if action == 'log':
                logger.warning(f"Alert: {alert_event['alert_name']} - {alert_event['current_value']}")
            elif action == 'email':
                # In production, send email notification
                logger.info(f"Email alert: {alert_event['alert_name']}")
            elif action == 'webhook':
                # In production, send webhook
                logger.info(f"Webhook alert: {alert_event['alert_name']}")
                
        except Exception as e:
            logger.error(f"Alert action execution failed: {e}")
            
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit analytics event"""
        try:
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    await handler(data)
        except Exception as e:
            logger.error(f"Event emission error: {e}")
            
    async def _create_default_dashboards(self):
        """Create default system dashboards"""
        try:
            # System overview dashboard
            system_dashboard = Dashboard(
                dashboard_id="system_overview",
                name="System Overview",
                description="High-level system metrics and health",
                widgets=[
                    {
                        'id': 'cpu_usage',
                        'type': 'gauge',
                        'title': 'CPU Usage',
                        'metric': 'system.cpu_usage',
                        'min_value': 0,
                        'max_value': 100,
                        'threshold': 80
                    },
                    {
                        'id': 'memory_usage',
                        'type': 'gauge',
                        'title': 'Memory Usage',
                        'metric': 'system.memory_usage',
                        'min_value': 0,
                        'max_value': 100,
                        'threshold': 85
                    },
                    {
                        'id': 'network_chart',
                        'type': 'chart',
                        'title': 'Network Traffic',
                        'metric': 'network.bytes_per_second'
                    }
                ]
            )
            
            await self.dashboard_manager.create_dashboard(system_dashboard)
            
            # Performance dashboard
            performance_dashboard = Dashboard(
                dashboard_id="performance_metrics",
                name="Performance Metrics",
                description="Application performance and response times",
                widgets=[
                    {
                        'id': 'response_time',
                        'type': 'chart',
                        'title': 'Response Time',
                        'metric': 'application.response_time'
                    },
                    {
                        'id': 'throughput',
                        'type': 'metric',
                        'title': 'Throughput',
                        'metric': 'application.requests_per_second'
                    },
                    {
                        'id': 'error_rate',
                        'type': 'gauge',
                        'title': 'Error Rate',
                        'metric': 'application.error_rate',
                        'min_value': 0,
                        'max_value': 100,
                        'threshold': 5
                    }
                ]
            )
            
            await self.dashboard_manager.create_dashboard(performance_dashboard)
            
            self.metrics['dashboard_count'] = len(self.dashboard_manager.dashboards)
            
        except Exception as e:
            logger.error(f"Failed to create default dashboards: {e}")
            
    async def _load_configurations(self):
        """Load analytics configurations from database"""
        try:
            # In production, load from database
            # For demo, create sample configurations
            
            # Sample alert
            cpu_alert = Alert(
                alert_id="cpu_high",
                name="High CPU Usage",
                description="CPU usage exceeds 80%",
                metric_name="system.cpu_usage",
                condition="> 80",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                time_window=TimeWindow.MINUTE,
                actions=["log", "email"]
            )
            
            await self.alert_manager.add_alert(cpu_alert)
            
            # Sample ML model
            cpu_model = MLModel(
                model_id="cpu_predictor",
                name="CPU Usage Predictor",
                model_type="linear_regression",
                target_metric="system.cpu_usage",
                features=["time_of_day", "day_of_week", "historical_values"]
            )
            
            await self.ml_predictor.create_model(cpu_model)
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            
    async def _setup_database(self):
        """Setup SQLite database for persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                metric_type TEXT NOT NULL,
                tags TEXT,
                timestamp REAL NOT NULL,
                source TEXT,
                unit TEXT
            );
            
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                metric_name TEXT NOT NULL,
                condition TEXT NOT NULL,
                threshold REAL NOT NULL,
                severity TEXT NOT NULL,
                time_window INTEGER NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                actions TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                last_triggered REAL,
                trigger_count INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS ml_models (
                model_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                target_metric TEXT NOT NULL,
                features TEXT NOT NULL,
                parameters TEXT,
                training_data_size INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0.0,
                last_trained REAL,
                enabled BOOLEAN DEFAULT TRUE
            );
            
            CREATE TABLE IF NOT EXISTS dashboards (
                dashboard_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                widgets TEXT NOT NULL,
                layout TEXT,
                filters TEXT,
                auto_refresh INTEGER DEFAULT 30,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                owner TEXT,
                shared BOOLEAN DEFAULT FALSE
            );
        """)
        
        conn.commit()
        conn.close()

# Global instance
_analytics_engine: Optional[AdvancedAnalyticsEngine] = None

async def initialize_analytics_engine(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the global analytics engine"""
    global _analytics_engine
    try:
        _analytics_engine = AdvancedAnalyticsEngine(config)
        return await _analytics_engine.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize analytics engine: {e}")
        return False

def get_analytics_engine() -> AdvancedAnalyticsEngine:
    """Get the global analytics engine instance"""
    global _analytics_engine
    if _analytics_engine is None:
        raise RuntimeError("Analytics engine not initialized. Call initialize_analytics_engine() first.")
    return _analytics_engine

async def shutdown_analytics_engine():
    """Shutdown the global analytics engine"""
    global _analytics_engine
    if _analytics_engine:
        await _analytics_engine.shutdown()
        _analytics_engine = None
