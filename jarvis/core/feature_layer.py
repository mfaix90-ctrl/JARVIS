# =============================================================================
# JARVIS v6.0.1 -- SESSION 04: FEATURE LAYER
# File:   jarvis/core/feature_layer.py
# Deps:   S03 (jarvis.core.data_layer) ONLY
# Layer:  Layer 2 -- Feature Preprocessing
# =============================================================================
# CONSTRAINTS ENFORCED:
#   - No numpy / scipy
#   - No logging module
#   - No datetime.now() / time()
#   - No random / uuid
#   - No file IO
#   - No S01 / S02 imports
#   - No state-layer / regime imports
#   - Deterministic only
#   - Exactly 99 features (FeatureDimensionError on violation)
#   - NaN/Inf -> 0.0 (nan_replaced_count / nan_replaced_names tracked)
#   - KS-test pure stdlib math
#   - VOLATILITY_SCALING defined exactly once here
#   - VolatilityScalingError if asset_class unknown
# =============================================================================

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from jarvis.core.data_layer import MarketData

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

FEATURE_VECTOR_SIZE: int = 99
DRIFT_WINDOW_MAX: int = 500
DRIFT_KS_ALPHA: float = 0.01          # p-value threshold -> trigger drift
DRIFT_HARD_STOP_SEVERITY: float = 0.8 # severity threshold for hard-stop count
DRIFT_HARD_STOP_RATIO: float = 0.10   # >10 % of features -> hard_stop = True

# VOLATILITY_SCALING -- defined exactly once in this module (FAS invariant)
# Maps asset_class -> annualised volatility scaling factor used in
# normalised-volatility-unit (NVU) feature computation.
VOLATILITY_SCALING: Dict[str, float] = {
    "crypto":      1.0,    # baseline; BTC/USDT reference
    "forex":       0.15,   # FX is ~15 % as volatile as crypto baseline
    "indices":     0.20,   # equity indices
    "commodities": 0.25,   # commodities
    "rates":       0.05,   # fixed income / rates
}

# Feature name catalogue (fixed order -> fixed dimension guarantee)
# Groups:
#   [00-14]  Price Action        (15)
#   [15-26]  Volume & Liquidity  (12)
#   [27-44]  Technical           (18)
#   [45-54]  Microstructure      (10)
#   [55-62]  Cross-Asset         (8)
#   [63-74]  On-Chain            (12)  -- optional; 0.0 when unavailable
#   [75-82]  Regime & State      (8)
#   [83-92]  Sentiment           (10)
#   [93-98]  Meta                (6)
# Total: 15+12+18+10+8+12+8+10+6 = 99
FEATURE_NAMES: List[str] = [
    # --- Price Action [0-14] ---
    "returns_1m",
    "returns_5m",
    "returns_15m",
    "returns_1h",
    "returns_4h",
    "volatility_5m",
    "volatility_1h",
    "trend_coherence_20",
    "trend_coherence_50",
    "trend_coherence_200",
    "support_distance",
    "resistance_distance",
    "fibonacci_level",
    "pivot_points",
    "range_position",
    # --- Volume & Liquidity [15-26] ---
    "volume_ma_ratio",
    "volume_std",
    "bid_ask_spread",
    "order_book_imbalance",
    "trade_intensity",
    "large_trades_count",
    "volume_price_correlation",
    "liquidity_score",
    "market_depth",
    "quote_spread",
    "effective_spread",
    "realized_spread",
    # --- Technical Indicators [27-44] ---
    "rsi_14",
    "rsi_28",
    "macd",
    "macd_signal",
    "macd_hist",
    "bollinger_upper",
    "bollinger_lower",
    "bollinger_width",
    "atr_14",
    "adx_14",
    "cci_20",
    "stoch_k",
    "stoch_d",
    "williams_r",
    "mfi_14",
    "obv",
    "cmf_20",
    "vwap_distance",
    # --- Microstructure [45-54] ---
    "tick_direction",
    "trade_classification",
    "order_flow_imbalance",
    "vpin",
    "micro_effective_spread",
    "price_impact",
    "kyle_lambda",
    "roll_spread",
    "adverse_selection_component",
    "market_quality_index",
    # --- Cross-Asset [55-62] ---
    "btc_eth_correlation",
    "btc_gold_spread",
    "btc_sp500_correlation",
    "crypto_index_performance",
    "sector_momentum",
    "alt_season_indicator",
    "dominance_btc",
    "stablecoin_flow",
    # --- On-Chain [63-74] ---
    "exchange_net_flow",
    "whale_transaction_count",
    "mvrv_ratio",
    "nvt_ratio",
    "active_addresses",
    "transaction_volume",
    "fees_total",
    "hash_rate",
    "mining_difficulty",
    "realized_cap",
    "utxo_age_distribution",
    "holder_composition",
    # --- Regime & State [75-82] ---
    "regime_hmm",
    "volatility_regime",
    "trend_regime",
    "stress_level",
    "crisis_probability",
    "regime_stability",
    "transition_probability",
    "regime_confidence",
    # --- Sentiment [83-92] ---
    "fear_greed_index",
    "sentiment_twitter",
    "sentiment_reddit",
    "news_sentiment",
    "google_trends",
    "funding_rate",
    "open_interest",
    "long_short_ratio",
    "liquidation_volume",
    "social_volume",
    # --- Meta [93-98] ---
    "data_quality_score",
    "feature_drift_score",
    "completeness_ratio",
    "staleness_indicator",
    "source_reliability",
    "prediction_confidence_lagged",
]

# Compile-time invariant check (evaluated once at module import)
assert len(FEATURE_NAMES) == FEATURE_VECTOR_SIZE, (
    f"FATAL: FEATURE_NAMES has {len(FEATURE_NAMES)} entries, "
    f"expected {FEATURE_VECTOR_SIZE}"
)

# ---------------------------------------------------------------------------
# CUSTOM EXCEPTIONS
# ---------------------------------------------------------------------------

class FeatureDimensionError(Exception):
    """Raised when the computed feature dict has != 99 entries."""


class VolatilityScalingError(Exception):
    """Raised when asset_class is not in VOLATILITY_SCALING."""


class NumericalInstabilityError(Exception):
    """Raised when a critical numeric field contains NaN or Inf."""


# ---------------------------------------------------------------------------
# ENUMS & DATACLASSES
# ---------------------------------------------------------------------------

class DriftAction(Enum):
    IGNORIEREN            = "ignorieren"
    LOGGEN                = "loggen"
    UNSICHERHEIT_ERHOEHEN = "unsicherheit_erhoehen"
    REKALIBRIEREN         = "rekalibrieren"   # reserved; never auto-triggered


@dataclass
class DriftResult:
    feature:      str
    ks_statistic: float
    p_value:      float
    severity:     float       # clipped to [0.0, 1.0]
    action:       DriftAction


@dataclass
class DriftSummary:
    results:        List[DriftResult] = field(default_factory=list)
    hard_stop:      bool = False
    hard_stop_ratio: float = 0.0   # fraction of features with severity >= 0.8


@dataclass
class FeatureResult:
    """Full output of FeatureLayer.compute_features()."""
    features:           Dict[str, float]   # exactly 99 entries
    nan_replaced_count: int
    nan_replaced_names: List[str]


# ---------------------------------------------------------------------------
# PURE-MATH HELPERS
# ---------------------------------------------------------------------------

def _safe_float(value: object) -> float:
    """Convert value to float; return 0.0 if NaN/Inf or conversion fails."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def _clip(value: float, lo: float, hi: float) -> float:
    """Deterministic clip without numpy."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _mean(data: List[float]) -> float:
    if not data:
        return 0.0
    return sum(data) / len(data)


def _std(data: List[float]) -> float:
    """Population standard deviation."""
    n = len(data)
    if n < 2:
        return 0.0
    mu = _mean(data)
    variance = sum((x - mu) ** 2 for x in data) / n
    if variance < 0.0:
        return 0.0
    return math.sqrt(variance)


def _sma(data: List[float], period: int) -> float:
    """Simple moving average over last `period` values."""
    if len(data) < period or period <= 0:
        return 0.0
    window = data[-period:]
    return sum(window) / len(window)


def _ema(data: List[float], period: int) -> float:
    """Exponential moving average (last value) -- pure iterative."""
    if not data or period <= 0:
        return 0.0
    k = 2.0 / (period + 1)
    ema = data[0]
    for price in data[1:]:
        ema = price * k + ema * (1.0 - k)
    return ema


def _sorted_copy(data: List[float]) -> List[float]:
    return sorted(data)


def _ecdf(sorted_data: List[float], x: float) -> float:
    """Empirical CDF at x for a pre-sorted list."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_data[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo / n


def _ks_statistic(sample_a: List[float], sample_b: List[float]) -> float:
    """
    Two-sample Kolmogorov-Smirnov statistic.
    Pure stdlib -- no numpy/scipy.
    Returns D in [0.0, 1.0].
    """
    if not sample_a or not sample_b:
        return 0.0
    sa = _sorted_copy(sample_a)
    sb = _sorted_copy(sample_b)
    # Evaluate at all unique values from both samples
    all_points = sa + sb
    d_max = 0.0
    for x in all_points:
        diff = abs(_ecdf(sa, x) - _ecdf(sb, x))
        if diff > d_max:
            d_max = diff
    return d_max


def _ks_p_value_approx(d: float, n1: int, n2: int) -> float:
    """
    Approximate p-value for the two-sample KS test using the
    Kolmogorov distribution asymptotic formula.
    p = 2 * sum_{k=1}^{inf} (-1)^{k+1} exp(-2 k^2 lambda^2)
    where lambda = d * sqrt(n1*n2 / (n1+n2)).
    Truncated at k=100 for deterministic convergence.
    Returns p in (0.0, 1.0].
    """
    if d <= 0.0 or n1 <= 0 or n2 <= 0:
        return 1.0
    n_eff = (n1 * n2) / (n1 + n2)
    if n_eff <= 0.0:
        return 1.0
    lam = d * math.sqrt(n_eff)
    p = 0.0
    for k in range(1, 101):
        term = ((-1) ** (k + 1)) * math.exp(-2.0 * (k * lam) ** 2)
        p += term
        if abs(term) < 1e-10:
            break
    p = 2.0 * p
    return _clip(p, 0.0, 1.0)


def _rsi(closes: List[float], period: int = 14) -> float:
    """RSI using Wilder smoothing. Returns value in [0, 100]."""
    if len(closes) < period + 1:
        return 50.0
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-delta)
    if len(gains) < period:
        return 50.0
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(highs: List[float], lows: List[float], closes: List[float],
         period: int = 14) -> float:
    """Average True Range."""
    n = min(len(highs), len(lows), len(closes))
    if n < 2:
        return 0.0
    trs: List[float] = []
    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if not trs:
        return 0.0
    window = trs[-period:] if len(trs) >= period else trs
    return sum(window) / len(window)


def _bollinger(closes: List[float], period: int = 20,
               num_std: float = 2.0) -> Tuple[float, float, float]:
    """
    Returns (upper, lower, width) of Bollinger Bands.
    width = (upper - lower) / mid  if mid != 0 else 0.0
    """
    if len(closes) < period:
        c = closes[-1] if closes else 0.0
        return c, c, 0.0
    window = closes[-period:]
    mid = sum(window) / period
    sigma = _std(window)
    upper = mid + num_std * sigma
    lower = mid - num_std * sigma
    width = (upper - lower) / mid if mid != 0.0 else 0.0
    return upper, lower, width


def _macd(closes: List[float],
          fast: int = 12, slow: int = 26,
          signal_period: int = 9) -> Tuple[float, float, float]:
    """Returns (macd_line, signal_line, histogram)."""
    if len(closes) < slow:
        return 0.0, 0.0, 0.0
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    # Approximate signal as EMA of macd over signal_period using last values
    # We compute macd at each bar to build a short history
    macd_history: List[float] = []
    for i in range(slow, len(closes) + 1):
        ef = _ema(closes[:i], fast)
        es = _ema(closes[:i], slow)
        macd_history.append(ef - es)
    if not macd_history:
        return macd_line, 0.0, macd_line
    signal = _ema(macd_history, signal_period)
    histogram = macd_line - signal
    return macd_line, signal, histogram


def _cci(highs: List[float], lows: List[float], closes: List[float],
         period: int = 20) -> float:
    """Commodity Channel Index."""
    n = min(len(highs), len(lows), len(closes))
    if n < period:
        return 0.0
    tp = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(n)]
    window_tp = tp[-period:]
    sma_tp = sum(window_tp) / period
    mean_dev = sum(abs(x - sma_tp) for x in window_tp) / period
    if mean_dev == 0.0:
        return 0.0
    return (tp[-1] - sma_tp) / (0.015 * mean_dev)


def _stochastic(highs: List[float], lows: List[float], closes: List[float],
                k_period: int = 14,
                d_period: int = 3) -> Tuple[float, float]:
    """Stochastic %K and %D."""
    n = min(len(highs), len(lows), len(closes))
    if n < k_period:
        return 50.0, 50.0
    k_values: List[float] = []
    for i in range(k_period - 1, n):
        h_max = max(highs[i - k_period + 1:i + 1])
        l_min = min(lows[i - k_period + 1:i + 1])
        denom = h_max - l_min
        if denom == 0.0:
            k_values.append(50.0)
        else:
            k_values.append(100.0 * (closes[i] - l_min) / denom)
    stoch_k = k_values[-1] if k_values else 50.0
    d_window = k_values[-d_period:] if len(k_values) >= d_period else k_values
    stoch_d = sum(d_window) / len(d_window) if d_window else 50.0
    return stoch_k, stoch_d


def _williams_r(highs: List[float], lows: List[float], closes: List[float],
                period: int = 14) -> float:
    """Williams %R in [-100, 0]."""
    n = min(len(highs), len(lows), len(closes))
    if n < period:
        return -50.0
    h_max = max(highs[-period:])
    l_min = min(lows[-period:])
    denom = h_max - l_min
    if denom == 0.0:
        return -50.0
    return -100.0 * (h_max - closes[-1]) / denom


def _obv(closes: List[float], volumes: List[float]) -> float:
    """On-Balance Volume (raw, not normalised)."""
    n = min(len(closes), len(volumes))
    if n < 2:
        return 0.0
    obv_val = 0.0
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv_val += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv_val -= volumes[i]
    return obv_val


def _mfi(highs: List[float], lows: List[float], closes: List[float],
         volumes: List[float], period: int = 14) -> float:
    """Money Flow Index in [0, 100]."""
    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n < period + 1:
        return 50.0
    pos_mf = 0.0
    neg_mf = 0.0
    for i in range(n - period, n):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        tp_prev = (highs[i - 1] + lows[i - 1] + closes[i - 1]) / 3.0
        mf = tp * volumes[i]
        if tp > tp_prev:
            pos_mf += mf
        else:
            neg_mf += mf
    if neg_mf == 0.0:
        return 100.0
    mfr = pos_mf / neg_mf
    return 100.0 - (100.0 / (1.0 + mfr))


def _cmf(highs: List[float], lows: List[float], closes: List[float],
         volumes: List[float], period: int = 20) -> float:
    """Chaikin Money Flow in [-1, 1]."""
    n = min(len(highs), len(lows), len(closes), len(volumes))
    if n < period:
        return 0.0
    mfv_sum = 0.0
    vol_sum = 0.0
    for i in range(n - period, n):
        hl = highs[i] - lows[i]
        if hl == 0.0:
            mfm = 0.0
        else:
            mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / hl
        mfv_sum += mfm * volumes[i]
        vol_sum += volumes[i]
    if vol_sum == 0.0:
        return 0.0
    return _clip(mfv_sum / vol_sum, -1.0, 1.0)


def _adx(highs: List[float], lows: List[float], closes: List[float],
         period: int = 14) -> float:
    """Average Directional Index in [0, 100]."""
    n = min(len(highs), len(lows), len(closes))
    if n < period + 1:
        return 0.0
    plus_dm: List[float] = []
    minus_dm: List[float] = []
    tr_list: List[float] = []
    for i in range(1, n):
        up_move   = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm.append(up_move   if up_move   > down_move and up_move   > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move   and down_move > 0 else 0.0)
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i]  - closes[i - 1]))
        tr_list.append(tr)

    def wilder_smooth(data: List[float], p: int) -> List[float]:
        if len(data) < p:
            return [0.0]
        smoothed = [sum(data[:p])]
        for i in range(p, len(data)):
            smoothed.append(smoothed[-1] - smoothed[-1] / p + data[i])
        return smoothed

    s_tr    = wilder_smooth(tr_list, period)
    s_plus  = wilder_smooth(plus_dm, period)
    s_minus = wilder_smooth(minus_dm, period)

    dx_list: List[float] = []
    for i in range(len(s_tr)):
        denom_tr = s_tr[i]
        if denom_tr == 0.0:
            continue
        di_plus  = 100.0 * s_plus[i]  / denom_tr
        di_minus = 100.0 * s_minus[i] / denom_tr
        denom_dx = di_plus + di_minus
        if denom_dx == 0.0:
            continue
        dx_list.append(100.0 * abs(di_plus - di_minus) / denom_dx)

    if not dx_list:
        return 0.0
    return sum(dx_list[-period:]) / min(len(dx_list), period)


def _trend_coherence(closes: List[float], period: int) -> float:
    """
    Fraction of consecutive up-moves in last `period` bars.
    Returns value in [0.0, 1.0].
    """
    n = len(closes)
    if n < period + 1 or period <= 0:
        return 0.5
    window = closes[-(period + 1):]
    up_count = sum(1 for i in range(1, len(window)) if window[i] > window[i - 1])
    return up_count / period


def _range_position(close: float, high: float, low: float) -> float:
    """Where is close within [low, high]. Returns [0.0, 1.0]."""
    span = high - low
    if span <= 0.0:
        return 0.5
    return _clip((close - low) / span, 0.0, 1.0)


def _vwap_distance(closes: List[float], volumes: List[float]) -> float:
    """Normalised distance of last close from VWAP. Returns signed ratio."""
    n = min(len(closes), len(volumes))
    if n == 0:
        return 0.0
    total_pv = sum(closes[i] * volumes[i] for i in range(n))
    total_v  = sum(volumes)
    if total_v == 0.0:
        return 0.0
    vwap = total_pv / total_v
    if vwap == 0.0:
        return 0.0
    return (closes[-1] - vwap) / vwap


def _roll_spread(closes: List[float]) -> float:
    """
    Roll (1984) implicit bid-ask spread estimate.
    spread = 2 * sqrt(-cov(delta_p_t, delta_p_{t-1}))
    Returns 0.0 if covariance is non-negative (spread undefined).
    """
    if len(closes) < 3:
        return 0.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    n = len(deltas) - 1
    if n <= 0:
        return 0.0
    mu_d = _mean(deltas)
    cov = sum((deltas[i] - mu_d) * (deltas[i + 1] - mu_d) for i in range(n)) / n
    if cov >= 0.0:
        return 0.0
    return 2.0 * math.sqrt(-cov)


def _kyle_lambda(price_changes: List[float], volumes: List[float]) -> float:
    """
    Kyle lambda: regression coefficient of abs(price_change) on volume.
    Returns 0.0 on degenerate input.
    """
    n = min(len(price_changes), len(volumes))
    if n < 2:
        return 0.0
    x = volumes[:n]
    y = [abs(price_changes[i]) for i in range(n)]
    mu_x = _mean(x)
    mu_y = _mean(y)
    cov_xy = sum((x[i] - mu_x) * (y[i] - mu_y) for i in range(n)) / n
    var_x  = sum((xi - mu_x) ** 2 for xi in x) / n
    if var_x == 0.0:
        return 0.0
    return cov_xy / var_x


# ---------------------------------------------------------------------------
# VOLATILITY SCALING HELPER
# ---------------------------------------------------------------------------

def get_volatility_scaling(asset_class: str) -> float:
    """
    Return the NVU volatility scaling factor for the given asset_class.
    Raises VolatilityScalingError if asset_class is unknown.
    """
    ac = asset_class.lower().strip()
    if ac not in VOLATILITY_SCALING:
        raise VolatilityScalingError(
            f"asset_class '{asset_class}' not in VOLATILITY_SCALING. "
            f"Known: {sorted(VOLATILITY_SCALING.keys())}"
        )
    return VOLATILITY_SCALING[ac]


# ---------------------------------------------------------------------------
# FEATURE COMPUTATION
# ---------------------------------------------------------------------------

class _FeatureComputer:
    """
    Internal stateless computation engine.
    All methods receive raw list data extracted from MarketData.
    All return values are finite floats; NaN/Inf handled by caller sanitiser.
    """

    # -- price action --------------------------------------------------------

    @staticmethod
    def _return(closes: List[float], lag: int) -> float:
        if len(closes) < lag + 1 or lag <= 0:
            return 0.0
        prev = closes[-(lag + 1)]
        curr = closes[-1]
        if prev == 0.0:
            return 0.0
        return (curr - prev) / prev

    @staticmethod
    def _volatility(closes: List[float], period: int) -> float:
        if len(closes) < period + 1:
            return 0.0
        rets = [(closes[i] - closes[i - 1]) / closes[i - 1]
                for i in range(len(closes) - period, len(closes))
                if closes[i - 1] != 0.0]
        return _std(rets) if rets else 0.0

    @staticmethod
    def _support_distance(closes: List[float], period: int = 20) -> float:
        if len(closes) < period:
            return 0.0
        window = closes[-period:]
        support = min(window)
        curr = closes[-1]
        if curr == 0.0:
            return 0.0
        return (curr - support) / curr

    @staticmethod
    def _resistance_distance(closes: List[float], period: int = 20) -> float:
        if len(closes) < period:
            return 0.0
        window = closes[-period:]
        resistance = max(window)
        curr = closes[-1]
        if curr == 0.0:
            return 0.0
        return (resistance - curr) / curr

    @staticmethod
    def _fibonacci_level(close: float, high: float, low: float) -> float:
        """
        Proximity to nearest Fibonacci retracement level.
        Returns [0.0, 1.0] where 1.0 = exactly on a fib level.
        Levels: 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0
        """
        span = high - low
        if span <= 0.0:
            return 0.0
        pos = _clip((close - low) / span, 0.0, 1.0)
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        min_dist = min(abs(pos - f) for f in fib_levels)
        return _clip(1.0 - min_dist / 0.5, 0.0, 1.0)

    @staticmethod
    def _pivot_points(high: float, low: float, close: float) -> float:
        """
        Classic pivot point normalised by close.
        pp = (high + low + close) / 3
        Returns (pp / close) - 1.0 as a signed ratio.
        """
        if close == 0.0:
            return 0.0
        pp = (high + low + close) / 3.0
        return (pp / close) - 1.0

    # -- volume & liquidity --------------------------------------------------

    @staticmethod
    def _volume_ma_ratio(volumes: List[float], period: int = 20) -> float:
        if len(volumes) < period + 1:
            return 1.0
        ma = _sma(volumes[:-1], period)
        if ma == 0.0:
            return 1.0
        return volumes[-1] / ma

    @staticmethod
    def _volume_price_correlation(closes: List[float],
                                  volumes: List[float],
                                  period: int = 20) -> float:
        n = min(len(closes), len(volumes))
        if n < period:
            return 0.0
        c = closes[-period:]
        v = volumes[-period:]
        mu_c = _mean(c)
        mu_v = _mean(v)
        cov  = sum((c[i] - mu_c) * (v[i] - mu_v) for i in range(period)) / period
        sd_c = _std(c)
        sd_v = _std(v)
        if sd_c == 0.0 or sd_v == 0.0:
            return 0.0
        return _clip(cov / (sd_c * sd_v), -1.0, 1.0)

    # -- technical computed fields ------------------------------------------
    # (delegates to module-level pure functions)


# ---------------------------------------------------------------------------
# MAIN FEATURE LAYER
# ---------------------------------------------------------------------------

class FeatureLayer:
    """
    SESSION 04 -- Feature Layer.

    Computes the fixed 99-dimensional feature vector from a MarketData object.
    Stateless with respect to history; history windows must be passed in via
    the `history` parameter or pre-loaded via push_history().

    Usage:
        layer = FeatureLayer(asset_class="crypto")
        layer.push_history(ohlcv_list)          # prime rolling windows
        result = layer.compute_features(market_data)

    Result.features has exactly 99 entries.  Any NaN/Inf produced during
    computation is replaced with 0.0 and tracked in nan_replaced_*.
    """

    def __init__(self, asset_class: str = "crypto") -> None:
        # Validate asset class immediately (VolatilityScalingError on unknown)
        self._volatility_scale: float = get_volatility_scaling(asset_class)
        self._asset_class: str = asset_class.lower().strip()

        # Rolling history windows (max DRIFT_WINDOW_MAX bars)
        self._opens:   deque = deque(maxlen=DRIFT_WINDOW_MAX)
        self._highs:   deque = deque(maxlen=DRIFT_WINDOW_MAX)
        self._lows:    deque = deque(maxlen=DRIFT_WINDOW_MAX)
        self._closes:  deque = deque(maxlen=DRIFT_WINDOW_MAX)
        self._volumes: deque = deque(maxlen=DRIFT_WINDOW_MAX)

        # Last quality_score from MarketData
        self._last_quality_score: float = 1.0

    # ------------------------------------------------------------------ API

    def push_history(self, ohlcv_records: List[Dict[str, float]]) -> None:
        """
        Prime rolling windows from a list of OHLCV dicts.
        Keys: 'open', 'high', 'low', 'close', 'volume'
        Earlier records first; last record becomes most recent bar.
        """
        for rec in ohlcv_records:
            self._opens.append(_safe_float(rec.get("open", 0.0)))
            self._highs.append(_safe_float(rec.get("high", 0.0)))
            self._lows.append(_safe_float(rec.get("low", 0.0)))
            self._closes.append(_safe_float(rec.get("close", 0.0)))
            self._volumes.append(_safe_float(rec.get("volume", 0.0)))

    def push_bar(self, open_: float, high: float, low: float,
                 close: float, volume: float) -> None:
        """Append a single OHLCV bar to the rolling windows."""
        self._opens.append(_safe_float(open_))
        self._highs.append(_safe_float(high))
        self._lows.append(_safe_float(low))
        self._closes.append(_safe_float(close))
        self._volumes.append(_safe_float(volume))

    def compute_features(self, market_data: MarketData) -> FeatureResult:
        """
        Compute exactly 99 features from market_data.

        The incoming bar is appended to the rolling windows before computation
        so that the result reflects the current bar.

        Returns FeatureResult with:
          - features: Dict[str, float] -- exactly 99 entries, all finite
          - nan_replaced_count: int
          - nan_replaced_names: List[str]

        Raises:
          FeatureDimensionError: if internal logic produces != 99 features
                                 (invariant violation -- should never happen)
        """
        ohlcv = market_data.ohlcv
        self.push_bar(ohlcv.open, ohlcv.high, ohlcv.low,
                      ohlcv.close, ohlcv.volume)
        self._last_quality_score = _safe_float(market_data.quality_score)

        raw = self._compute_raw()
        return self._sanitise(raw)

    # ------------------------------------------------------------------ INTERNAL

    def _compute_raw(self) -> Dict[str, float]:
        """
        Compute all 99 features; values may be NaN/Inf at this stage.
        Sanitisation is done by the caller (_sanitise).
        """
        closes  = list(self._closes)
        opens   = list(self._opens)
        highs   = list(self._highs)
        lows    = list(self._lows)
        volumes = list(self._volumes)

        n = len(closes)
        last_c  = closes[-1]  if closes  else 0.0
        last_h  = highs[-1]   if highs   else 0.0
        last_l  = lows[-1]    if lows    else 0.0
        last_v  = volumes[-1] if volumes else 0.0
        vs      = self._volatility_scale

        # --- Price Action [0-14] ---
        ret_1m  = _FeatureComputer._return(closes, 1)
        ret_5m  = _FeatureComputer._return(closes, 5)
        ret_15m = _FeatureComputer._return(closes, 15)
        ret_1h  = _FeatureComputer._return(closes, 60)
        ret_4h  = _FeatureComputer._return(closes, 240)

        vol_5m  = _FeatureComputer._volatility(closes, 5)  * vs
        vol_1h  = _FeatureComputer._volatility(closes, 60) * vs

        tc_20   = _trend_coherence(closes, 20)
        tc_50   = _trend_coherence(closes, 50)
        tc_200  = _trend_coherence(closes, 200)

        sup_dist = _FeatureComputer._support_distance(closes)
        res_dist = _FeatureComputer._resistance_distance(closes)
        fib      = _FeatureComputer._fibonacci_level(last_c, last_h, last_l)
        pivots   = _FeatureComputer._pivot_points(last_h, last_l, last_c)
        range_p  = _range_position(last_c, last_h, last_l)

        # --- Volume & Liquidity [15-26] ---
        vol_ma_ratio = _FeatureComputer._volume_ma_ratio(volumes)
        vol_std      = _std(volumes[-20:]) if len(volumes) >= 20 else _std(volumes)
        # bid_ask_spread, order_book_imbalance, trade_intensity, large_trades,
        # liquidity_score, market_depth, quote_spread -- not in OHLCV; proxy from ATR
        atr_val      = _atr(highs, lows, closes, 14) if n >= 2 else 0.0
        bid_ask_est  = atr_val / last_c if last_c != 0.0 else 0.0
        obi          = 0.0   # order book imbalance -- unavailable without L2
        trade_int    = last_v / (volumes[-20:].__len__() or 1)
        large_trades = 0.0   # unavailable without tick data
        vpc          = _FeatureComputer._volume_price_correlation(closes, volumes)
        liq_score    = _clip(1.0 - bid_ask_est, 0.0, 1.0)
        mkt_depth    = 0.0   # unavailable without L2
        quote_spr    = bid_ask_est
        eff_spread   = bid_ask_est * 0.5
        real_spread  = bid_ask_est * 0.3

        # --- Technical Indicators [27-44] ---
        rsi14  = _rsi(closes, 14)
        rsi28  = _rsi(closes, 28)
        macd_l, macd_s, macd_h = _macd(closes)
        bb_up, bb_lo, bb_w     = _bollinger(closes)
        atr14  = atr_val
        adx14  = _adx(highs, lows, closes, 14)
        cci20  = _cci(highs, lows, closes, 20)
        sk, sd = _stochastic(highs, lows, closes)
        wr     = _williams_r(highs, lows, closes)
        mfi14  = _mfi(highs, lows, closes, volumes, 14)
        obv_v  = _obv(closes, volumes)
        cmf20  = _cmf(highs, lows, closes, volumes, 20)
        vwap_d = _vwap_distance(closes, volumes)

        # Normalise obv to avoid extreme raw values
        vol_sum = sum(volumes) if volumes else 1.0
        obv_norm = obv_v / vol_sum if vol_sum != 0.0 else 0.0

        # --- Microstructure [45-54] ---
        # tick_direction: +1 uptick, -1 downtick, 0 unchanged (last bar)
        if n >= 2:
            prev_c = closes[-2]
            tick_dir = 1.0 if last_c > prev_c else (-1.0 if last_c < prev_c else 0.0)
        else:
            tick_dir = 0.0
        trade_class  = tick_dir             # simplified Lee-Ready proxy
        ofi          = 0.0                  # order flow imbalance -- needs L2
        vpin_v       = 0.0                  # VPIN -- needs tick data
        micro_eff_sp = eff_spread
        price_impact = kyle_l = _kyle_lambda(
            [closes[i] - closes[i - 1] for i in range(1, n)] if n >= 2 else [0.0],
            volumes[1:n] if n >= 2 else [1.0],
        )
        roll_sp      = _roll_spread(closes)
        adv_sel      = 0.0                  # adverse selection -- needs L2
        mqi          = liq_score            # market quality proxy

        # --- Cross-Asset [55-62] ---
        # All cross-asset values require external data not present in
        # single-symbol MarketData.  Set to 0.0 (feature mask = inactive).
        btc_eth_corr = 0.0
        btc_gold_spr = 0.0
        btc_sp5_corr = 0.0
        cip          = 0.0
        sec_mom      = 0.0
        alt_season   = 0.0
        dom_btc      = 0.0
        stable_flow  = 0.0

        # --- On-Chain [63-74] ---
        # Optional; 0.0 when unavailable (feature mask tracks this).
        exc_net_flow  = 0.0
        whale_cnt     = 0.0
        mvrv          = 0.0
        nvt           = 0.0
        active_addr   = 0.0
        tx_vol        = 0.0
        fees          = 0.0
        hash_r        = 0.0
        mining_diff   = 0.0
        real_cap      = 0.0
        utxo_age      = 0.0
        holder_comp   = 0.0

        # --- Regime & State [75-82] ---
        # S04 does not import S05 (regime).  Proxy from OHLCV volatility.
        regime_hmm_v   = 0.0   # requires S05; placeholder
        vol_reg        = _clip(vol_1h, 0.0, 1.0)
        trend_reg      = tc_50
        stress_lv      = _clip(vol_1h * 2.0, 0.0, 1.0)
        crisis_prob    = _clip(vol_1h * 3.0, 0.0, 1.0)
        reg_stab       = 1.0 - stress_lv
        trans_prob     = 0.0   # requires S05
        reg_conf       = 1.0 - stress_lv

        # --- Sentiment [83-92] ---
        # External data unavailable; 0.0 / neutral defaults
        fg_idx         = market_data.features.get("fear_greed_index", 0.5)
        sent_tw        = market_data.features.get("sentiment_twitter",  0.0)
        sent_rd        = market_data.features.get("sentiment_reddit",   0.0)
        news_sent      = market_data.features.get("news_sentiment",     0.0)
        g_trends       = market_data.features.get("google_trends",      0.0)
        fund_rate      = market_data.features.get("funding_rate",       0.0)
        oi             = market_data.features.get("open_interest",      0.0)
        ls_ratio       = market_data.features.get("long_short_ratio",   1.0)
        liq_vol        = market_data.features.get("liquidation_volume", 0.0)
        soc_vol        = market_data.features.get("social_volume",      0.0)

        # --- Meta [93-98] ---
        dq_score       = self._last_quality_score
        feat_drift_sc  = 0.0   # populated after FeatureDriftMonitor.scan()
        completeness   = self._compute_completeness(market_data)
        staleness      = 0.0   # set externally via MarketData.is_stale if available
        src_reliability = dq_score
        pred_conf_lag  = market_data.features.get("prediction_confidence_lagged", 0.0)

        # Assemble in FEATURE_NAMES order
        raw: Dict[str, float] = {
            # Price Action
            "returns_1m":            ret_1m,
            "returns_5m":            ret_5m,
            "returns_15m":           ret_15m,
            "returns_1h":            ret_1h,
            "returns_4h":            ret_4h,
            "volatility_5m":         vol_5m,
            "volatility_1h":         vol_1h,
            "trend_coherence_20":    tc_20,
            "trend_coherence_50":    tc_50,
            "trend_coherence_200":   tc_200,
            "support_distance":      sup_dist,
            "resistance_distance":   res_dist,
            "fibonacci_level":       fib,
            "pivot_points":          pivots,
            "range_position":        range_p,
            # Volume & Liquidity
            "volume_ma_ratio":       vol_ma_ratio,
            "volume_std":            vol_std,
            "bid_ask_spread":        bid_ask_est,
            "order_book_imbalance":  obi,
            "trade_intensity":       trade_int,
            "large_trades_count":    large_trades,
            "volume_price_correlation": vpc,
            "liquidity_score":       liq_score,
            "market_depth":          mkt_depth,
            "quote_spread":          quote_spr,
            "effective_spread":      eff_spread,
            "realized_spread":       real_spread,
            # Technical Indicators
            "rsi_14":                rsi14,
            "rsi_28":                rsi28,
            "macd":                  macd_l,
            "macd_signal":           macd_s,
            "macd_hist":             macd_h,
            "bollinger_upper":       bb_up,
            "bollinger_lower":       bb_lo,
            "bollinger_width":       bb_w,
            "atr_14":                atr14,
            "adx_14":                adx14,
            "cci_20":                cci20,
            "stoch_k":               sk,
            "stoch_d":               sd,
            "williams_r":            wr,
            "mfi_14":                mfi14,
            "obv":                   obv_norm,
            "cmf_20":                cmf20,
            "vwap_distance":         vwap_d,
            # Microstructure
            "tick_direction":        tick_dir,
            "trade_classification":  trade_class,
            "order_flow_imbalance":  ofi,
            "vpin":                  vpin_v,
            "micro_effective_spread": micro_eff_sp,
            "price_impact":          price_impact,
            "kyle_lambda":           kyle_l,
            "roll_spread":           roll_sp,
            "adverse_selection_component": adv_sel,
            "market_quality_index":  mqi,
            # Cross-Asset
            "btc_eth_correlation":       btc_eth_corr,
            "btc_gold_spread":           btc_gold_spr,
            "btc_sp500_correlation":     btc_sp5_corr,
            "crypto_index_performance":  cip,
            "sector_momentum":           sec_mom,
            "alt_season_indicator":      alt_season,
            "dominance_btc":             dom_btc,
            "stablecoin_flow":           stable_flow,
            # On-Chain
            "exchange_net_flow":         exc_net_flow,
            "whale_transaction_count":   whale_cnt,
            "mvrv_ratio":                mvrv,
            "nvt_ratio":                 nvt,
            "active_addresses":          active_addr,
            "transaction_volume":        tx_vol,
            "fees_total":                fees,
            "hash_rate":                 hash_r,
            "mining_difficulty":         mining_diff,
            "realized_cap":              real_cap,
            "utxo_age_distribution":     utxo_age,
            "holder_composition":        holder_comp,
            # Regime & State
            "regime_hmm":                regime_hmm_v,
            "volatility_regime":         vol_reg,
            "trend_regime":              trend_reg,
            "stress_level":              stress_lv,
            "crisis_probability":        crisis_prob,
            "regime_stability":          reg_stab,
            "transition_probability":    trans_prob,
            "regime_confidence":         reg_conf,
            # Sentiment
            "fear_greed_index":          _safe_float(fg_idx),
            "sentiment_twitter":         _safe_float(sent_tw),
            "sentiment_reddit":          _safe_float(sent_rd),
            "news_sentiment":            _safe_float(news_sent),
            "google_trends":             _safe_float(g_trends),
            "funding_rate":              _safe_float(fund_rate),
            "open_interest":             _safe_float(oi),
            "long_short_ratio":          _safe_float(ls_ratio),
            "liquidation_volume":        _safe_float(liq_vol),
            "social_volume":             _safe_float(soc_vol),
            # Meta
            "data_quality_score":            dq_score,
            "feature_drift_score":           feat_drift_sc,
            "completeness_ratio":            completeness,
            "staleness_indicator":           staleness,
            "source_reliability":            src_reliability,
            "prediction_confidence_lagged":  _safe_float(pred_conf_lag),
        }
        return raw

    @staticmethod
    def _sanitise(raw: Dict[str, float]) -> FeatureResult:
        """
        Replace any NaN/Inf with 0.0.
        Enforce exactly FEATURE_VECTOR_SIZE entries (by ordered FEATURE_NAMES).
        Raises FeatureDimensionError if raw does not contain all 99 names.
        """
        missing = [n for n in FEATURE_NAMES if n not in raw]
        if missing:
            raise FeatureDimensionError(
                f"Feature computation missing {len(missing)} feature(s): {missing}"
            )

        nan_names: List[str] = []
        clean: Dict[str, float] = {}

        for name in FEATURE_NAMES:
            v = raw[name]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            if math.isnan(fv) or math.isinf(fv):
                nan_names.append(name)
                clean[name] = 0.0
            else:
                clean[name] = fv

        # Final dimension guard
        if len(clean) != FEATURE_VECTOR_SIZE:
            raise FeatureDimensionError(
                f"Sanitised feature dict has {len(clean)} entries, "
                f"expected {FEATURE_VECTOR_SIZE}"
            )

        return FeatureResult(
            features=clean,
            nan_replaced_count=len(nan_names),
            nan_replaced_names=nan_names,
        )

    @staticmethod
    def _compute_completeness(market_data: MarketData) -> float:
        """Fraction of expected external feature keys that are present and finite."""
        expected_keys = [
            "fear_greed_index", "sentiment_twitter", "sentiment_reddit",
            "news_sentiment", "google_trends", "funding_rate",
            "open_interest", "long_short_ratio", "liquidation_volume",
            "social_volume", "prediction_confidence_lagged",
        ]
        present = 0
        for k in expected_keys:
            v = market_data.features.get(k, None)
            if v is not None:
                try:
                    fv = float(v)
                    if not (math.isnan(fv) or math.isinf(fv)):
                        present += 1
                except (TypeError, ValueError):
                    pass
        return present / len(expected_keys)


# ---------------------------------------------------------------------------
# FEATURE DRIFT MONITOR
# ---------------------------------------------------------------------------

class FeatureDriftMonitor:
    """
    Monitors distribution drift for each feature via rolling windows and
    two-sample KS-test (pure stdlib).

    Usage:
        monitor = FeatureDriftMonitor()
        monitor.update(feature_result.features)   # call each compute cycle
        drift_summary = monitor.scan()            # check all features
    """

    def __init__(self, window: int = DRIFT_WINDOW_MAX) -> None:
        if window < 1 or window > DRIFT_WINDOW_MAX:
            raise ValueError(
                f"window must be in [1, {DRIFT_WINDOW_MAX}], got {window}"
            )
        self._window: int = window
        # Each feature has a deque of historical values (reference distribution).
        # defaultdict allows direct test-time injection via _history[name].append()
        # for feature names not in FEATURE_NAMES (edge-case testing).
        # detect_drift() raises KeyError if a feature was never accessed.
        _maxlen: int = window
        self._history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=_maxlen)  # type: ignore[return-value]
        )
        # Pre-populate all canonical feature names so they are always present.
        for name in FEATURE_NAMES:
            _ = self._history[name]  # touch to create
        # Sliding "current" window (half of history used as recent sample)
        self._recent_len: int = max(1, window // 2)

    # ------------------------------------------------------------------ API

    def update(self, features: Dict[str, float]) -> None:
        """
        Append current feature values to rolling history windows.
        Call once per compute cycle after FeatureLayer.compute_features().
        Ignores unknown feature keys (strict subset only).
        """
        for name in FEATURE_NAMES:
            if name in features:
                self._history[name].append(features[name])

    def detect_drift(self, feature: str,
                     window: int = DRIFT_WINDOW_MAX) -> DriftResult:
        """
        Compute KS-test drift for a single feature.

        The full history is split into:
          reference : first half of available observations
          recent    : last half of available observations
        Returns DriftResult with ks_statistic, p_value, severity, action.
        """
        if feature not in self._history:
            raise KeyError(f"Unknown feature: '{feature}'")

        hist = list(self._history[feature])
        n = len(hist)

        if n < 4:
            # Not enough data for meaningful test
            return DriftResult(
                feature=feature,
                ks_statistic=0.0,
                p_value=1.0,
                severity=0.0,
                action=DriftAction.IGNORIEREN,
            )

        split = n // 2
        reference = hist[:split]
        recent    = hist[split:]

        d = _ks_statistic(reference, recent)
        p = _ks_p_value_approx(d, len(reference), len(recent))

        # Severity: KS statistic weighted by inverse p-value influence
        # higher d and lower p -> higher severity
        p_factor = 1.0 - p          # 0 when p=1 (no drift), 1 when p=0
        severity = _clip(d * p_factor, 0.0, 1.0)

        action = self.determine_action(severity)

        return DriftResult(
            feature=feature,
            ks_statistic=d,
            p_value=p,
            severity=severity,
            action=action,
        )

    def scan(self) -> DriftSummary:
        """
        Run detect_drift for all 99 features.
        Computes hard_stop flag: True if >10% of features have severity >= 0.8.
        Does NOT raise an exception for hard_stop.
        """
        results: List[DriftResult] = []
        for name in FEATURE_NAMES:
            results.append(self.detect_drift(name))

        high_severity_count = sum(
            1 for r in results
            if r.severity >= DRIFT_HARD_STOP_SEVERITY
        )
        ratio = high_severity_count / FEATURE_VECTOR_SIZE
        hard_stop = ratio > DRIFT_HARD_STOP_RATIO

        return DriftSummary(
            results=results,
            hard_stop=hard_stop,
            hard_stop_ratio=ratio,
        )

    def calculate_severity(self, ks_stat: float,
                            importance: float) -> float:
        """
        Compute severity from KS statistic and feature importance weight.

        Rules:
          - If ks_stat or importance is NaN -> return 1.0 (max severity)
          - Otherwise: severity = clip(ks_stat * importance, 0.0, 1.0)
        """
        if math.isnan(ks_stat) or math.isnan(importance):
            return 1.0
        return _clip(ks_stat * importance, 0.0, 1.0)

    @staticmethod
    def determine_action(severity: float) -> DriftAction:
        """
        Map severity to DriftAction (REKALIBRIEREN never auto-triggered).

          [0.0, 0.2)   -> IGNORIEREN
          [0.2, 0.5)   -> LOGGEN       (signal only, no I/O)
          [0.5, 1.0]   -> UNSICHERHEIT_ERHOEHEN
        """
        if severity < 0.2:
            return DriftAction.IGNORIEREN
        if severity < 0.5:
            return DriftAction.LOGGEN
        return DriftAction.UNSICHERHEIT_ERHOEHEN

    def trigger_response(self, drift: DriftResult) -> DriftAction:
        """Return the action encoded in the DriftResult (no side effects)."""
        return drift.action
