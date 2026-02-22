# jarvis/core/__init__.py
# Core canonical types for the JARVIS platform.
# Authoritative import source: jarvis.core.regime

from jarvis.core.integrity_layer import IntegrityLayer
from jarvis.core.logging_layer import EventLogger, Event, EventFilter
from jarvis.core.data_layer import (
    OHLCV,
    MarketData,
    EnhancedMarketData,
    ValidationResult,
    DataCache,
    NumericalInstabilityError,
    DataQualityError,
    SequenceError,
)
from jarvis.core.feature_layer import (
    FeatureLayer,
    FeatureDriftMonitor,
    DriftResult,
    DriftSummary,
    DriftAction,
    FeatureDimensionError,
    VolatilityScalingError,
)
from jarvis.core.execution_guard import (
    build_execution_order,
    ExecutionOrder,
)
