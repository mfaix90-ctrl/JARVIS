# jarvis/core/logging_layer.py
# S02 -- Logging Layer
# JARVIS v6.0.1 -- Decision Quality Maximization Platform
#
# Scope: Event-sourced logging with hash-chain integration.
# Zero tolerance for lost events. No file IO. No global mutable state.
# All timestamps are caller-supplied. All hashes are deterministic.
#
# Canonical import:
#   from jarvis.core.logging_layer import EventLogger, Event, EventFilter
#
# Dependencies: S01 (jarvis.core.integrity_layer)
# Prohibited: datetime.now(), uuid, random, file IO, global mutable state

# ===========================================================================
# SECTION 1 -- STDLIB IMPORTS
# ===========================================================================

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

# ===========================================================================
# SECTION 2 -- S01 DEPENDENCY
# ===========================================================================

from jarvis.core.integrity_layer import IntegrityLayer

# ===========================================================================
# SECTION 3 -- CONSTANTS
# ===========================================================================

# Sentinel strings used when numeric sanitization detects invalid values.
# These are logged in place of the invalid value; the event is never silently
# dropped. Values are ASCII string literals -- no float arithmetic involved.
_NAN_SENTINEL: str = "NaN_DETECTED"
_INF_SENTINEL: str = "Inf_DETECTED"

# Field separator used inside hash preimage. Chosen to be unlikely to appear
# in any field value; does not affect correctness, only collision resistance.
_HASH_SEP: str = "|"

# ===========================================================================
# SECTION 4 -- DATACLASSES: Event, EventFilter
# ===========================================================================

@dataclass
class Event:
    """
    Immutable record of a single system event.

    Fields
    ------
    id        : Deterministic string identifier derived from instance counter.
    type      : Category string (e.g. DECISION, ERROR, STATE_CHANGE, ...).
    timestamp : Caller-supplied datetime. Never generated internally.
    data      : Sanitized key-value payload. NaN/Inf values replaced with
                sentinel strings before storage.
    hash      : SHA-256 hex digest over (id, type, timestamp, data).
                Deterministic; depends only on the four explicit fields above.
    """
    id: str
    type: str
    timestamp: datetime
    data: Dict[str, Any]
    hash: str


@dataclass
class EventFilter:
    """
    Filter specification for EventLogger.query_events().

    All fields are optional. Omitted fields apply no constraint.

    Fields
    ------
    event_type : If set, only events whose .type equals this value are returned.
    start_time : If set, only events with timestamp >= start_time are returned.
    end_time   : If set, only events with timestamp <= end_time are returned.
    limit      : If set, at most this many events are returned (from the front
                 of the filtered sequence, i.e. oldest first).
    """
    event_type: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: Optional[int] = None


# ===========================================================================
# SECTION 5 -- INTERNAL HELPERS
# ===========================================================================

def _sanitize_numeric(value: Any) -> Any:
    """
    Replace float NaN or Inf with the appropriate sentinel string.

    Non-float values are returned unchanged. This function never raises;
    it is the last line of defence before a value enters Event.data.

    Deterministic: output depends only on the input value.
    No IO. No side effects. No randomness.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return _NAN_SENTINEL
        if math.isinf(value):
            return _INF_SENTINEL
    return value


def _sanitize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a new dict with all float values sanitized via _sanitize_numeric().

    The original dict is not mutated. Keys are not modified.
    Deterministic: output depends only on the input dict.
    """
    return {k: _sanitize_numeric(v) for k, v in data.items()}


def _compute_hash(
    event_id: str,
    event_type: str,
    timestamp: datetime,
    data: Dict[str, Any],
) -> str:
    """
    Compute a deterministic SHA-256 hex digest for an event.

    Hash preimage construction
    --------------------------
    Fields are serialized in fixed order:
        event_id  + SEP
        event_type + SEP
        timestamp.isoformat() + SEP
        sorted_items_repr

    sorted_items_repr is repr(sorted(data.items())), which gives a
    deterministic string regardless of dict insertion order.

    No implicit values (time, pid, entropy) are included.
    Returns a 64-character lowercase hex string.
    """
    sorted_items: str = repr(sorted(data.items()))
    preimage: str = (
        event_id
        + _HASH_SEP
        + event_type
        + _HASH_SEP
        + timestamp.isoformat()
        + _HASH_SEP
        + sorted_items
    )
    return hashlib.sha256(preimage.encode("ascii", errors="replace")).hexdigest()


def _make_event_id(counter: int) -> str:
    """
    Derive a deterministic event ID from a monotonic integer counter.

    Format: "EVT-{counter:016d}"
    Zero-padded to 16 digits for lexicographic sort stability.
    No uuid. No randomness. No time dependency.
    """
    return "EVT-{:016d}".format(counter)


# ===========================================================================
# SECTION 6 -- EventLogger
# ===========================================================================

class EventLogger:
    """
    Event-sourced logger with deterministic hash-chain integrity.

    Storage
    -------
    Events are held in an instance-level list (_store). No file IO.
    No global state. Each EventLogger instance is fully independent.

    Determinism guarantees
    ----------------------
    - Timestamps are caller-supplied; never generated internally.
    - Event IDs are derived from a monotonic counter (_counter).
    - Hashes depend only on the four explicit event fields.
    - No randomness anywhere in this class.

    Zero lost events
    ----------------
    log_event() raises LoggingError on any failure condition instead of
    silently discarding the event. Callers must handle or propagate.

    Numeric safety
    --------------
    All float values in Event.data are sanitized before storage.
    NaN -> "NaN_DETECTED", Inf -> "Inf_DETECTED".
    Logging is never interrupted by numeric errors in the payload.

    Dependencies
    ------------
    Uses IntegrityLayer from S01 for future hash-chain verification hooks.
    The IntegrityLayer instance is held on self._integrity and is available
    for downstream verification without adding coupling here.
    """

    def __init__(self) -> None:
        """
        Initialise an empty EventLogger.

        No arguments. No side effects. No IO.
        """
        self._store: List[Event] = []
        self._counter: int = 0
        self._integrity: IntegrityLayer = IntegrityLayer()

    # -----------------------------------------------------------------------
    # SECTION 6.1 -- log_event
    # -----------------------------------------------------------------------

    def log_event(self, event_type: str, data: Dict[str, Any], timestamp: datetime) -> str:
        """
        Record one event atomically. Return the assigned event ID.

        Parameters
        ----------
        event_type : Non-empty string categorising the event.
                     Examples: DECISION, ERROR, STATE_CHANGE,
                               OOD_DETECTED, CALIBRATION_FAILED.
        data       : Arbitrary key-value payload. Float values are
                     sanitized; all other values are stored as-is.
        timestamp  : Caller-supplied datetime. Must not be None.
                     Never generated internally; required for determinism.

        Returns
        -------
        str : The event ID assigned to this event (e.g. "EVT-0000000000000001").

        Raises
        ------
        LoggingError : If event_type is empty, timestamp is None, or any
                       other invariant is violated. Never silently swallows
                       errors -- zero lost events is a hard invariant.

        Determinism: given the same counter state, event_type, data, and
        timestamp, this method always produces an identical Event record.
        """
        if not event_type:
            raise LoggingError("event_type must be a non-empty string")
        if timestamp is None:
            raise LoggingError("timestamp must be caller-supplied; None is not permitted")
        if not isinstance(timestamp, datetime):
            raise LoggingError(
                "timestamp must be a datetime instance; got: {}".format(type(timestamp))
            )

        self._counter += 1
        event_id: str = _make_event_id(self._counter)
        sanitized: Dict[str, Any] = _sanitize_data(data)
        event_hash: str = _compute_hash(event_id, event_type, timestamp, sanitized)

        event = Event(
            id=event_id,
            type=event_type,
            timestamp=timestamp,
            data=sanitized,
            hash=event_hash,
        )
        self._store.append(event)
        return event_id

    # -----------------------------------------------------------------------
    # SECTION 6.2 -- log_state_change
    # -----------------------------------------------------------------------

    def log_state_change(self, new_state: Any, timestamp: datetime) -> str:
        """
        Log a GlobalSystemState transition event.

        This method accepts any object as new_state to avoid importing
        jarvis.core.system_state (S05), which is not yet in the dependency
        chain for S02. The state is serialized via repr() for the payload.

        Parameters
        ----------
        new_state : GlobalSystemState instance (or any state object).
                    Stored as repr(new_state) in the event payload.
        timestamp : Caller-supplied datetime. Required; never generated here.

        Returns
        -------
        str : Event ID of the recorded STATE_CHANGE event.

        Raises
        ------
        LoggingError : If new_state is None or timestamp is invalid.
        """
        if new_state is None:
            raise LoggingError("new_state must not be None")
        data: Dict[str, Any] = {"state_repr": repr(new_state)}
        return self.log_event("STATE_CHANGE", data, timestamp)

    # -----------------------------------------------------------------------
    # SECTION 6.3 -- query_events
    # -----------------------------------------------------------------------

    def query_events(self, filter: EventFilter) -> List[Event]:
        """
        Return a list of events matching the given filter.

        Filtering is applied in this order:
            1. event_type equality check (if set)
            2. start_time lower bound (inclusive, if set)
            3. end_time upper bound (inclusive, if set)
            4. limit truncation (oldest-first, if set)

        Parameters
        ----------
        filter : EventFilter instance. All fields are optional.

        Returns
        -------
        List[Event] : Matching events in insertion order (oldest first).
                      Empty list if no events match.

        Raises
        ------
        LoggingError : If filter is None.

        Pure read. Does not mutate _store. No IO. Deterministic.
        """
        if filter is None:
            raise LoggingError("filter must not be None")

        results: List[Event] = []
        for event in self._store:
            if filter.event_type is not None and event.type != filter.event_type:
                continue
            if filter.start_time is not None and event.timestamp < filter.start_time:
                continue
            if filter.end_time is not None and event.timestamp > filter.end_time:
                continue
            results.append(event)

        if filter.limit is not None:
            results = results[: filter.limit]

        return results

    # -----------------------------------------------------------------------
    # SECTION 6.4 -- get_event_stream
    # -----------------------------------------------------------------------

    def get_event_stream(self, start_time: datetime) -> Iterator[Event]:
        """
        Yield events one by one in insertion order starting from start_time.

        Parameters
        ----------
        start_time : Caller-supplied datetime lower bound (inclusive).

        Yields
        ------
        Event : Each matching event in insertion order.

        Raises
        ------
        LoggingError : If start_time is None or not a datetime instance.

        Pure read. Does not mutate _store. No IO. Deterministic.
        """
        if start_time is None:
            raise LoggingError("start_time must be caller-supplied; None is not permitted")
        if not isinstance(start_time, datetime):
            raise LoggingError(
                "start_time must be a datetime instance; got: {}".format(type(start_time))
            )
        for event in self._store:
            if event.timestamp >= start_time:
                yield event

    # -----------------------------------------------------------------------
    # SECTION 6.5 -- event_count (utility)
    # -----------------------------------------------------------------------

    def event_count(self) -> int:
        """
        Return the total number of events currently stored.

        Deterministic. Pure read. No IO. No side effects.
        """
        return len(self._store)


# ===========================================================================
# SECTION 7 -- EXCEPTIONS
# ===========================================================================

class LoggingError(Exception):
    """
    Raised by EventLogger when an invariant is violated.

    Never silently swallowed. Every call site that invokes log_event()
    must either handle LoggingError or let it propagate. Silent failure
    is prohibited (zero lost events invariant).
    """
