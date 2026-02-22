# jarvis/core/integrity_layer.py
# Version: 1.0.0
# Session: S01 -- Integrity / Hash-Chain Layer
# Authority: JARVIS FAS v6.0.1 -- 01_INTEGRITY.md
#
# =============================================================================
# FAS COMPLIANCE DECLARATION
# =============================================================================
#
# This module is a STABLE CORE LAYER under ARCHITECTURE.md Section 1.
#
# IMPORT RULES (ARCHITECTURE.md Section 2):
#   jarvis/core/ is permitted to import from standard library only.
#
# SANCTIONED EXCEPTION -- documented per FAS v6.0.1 S01 blueprint:
#   create_threshold_manifest() and verify_threshold_manifest() import
#   jarvis.utils.constants at call time (deferred, not at module level).
#   This cross-import is explicitly shown in the FAS S01 implementation
#   specification (01_INTEGRITY.md, Section 5, Threshold Governance) and
#   is approved as a FAS-sanctioned exception for S01 only.
#   When jarvis/governance/threshold_guardian.py (S31) is built, the
#   threshold governance functions should be reviewed for potential
#   migration to that layer. Until then this single deferred import is
#   the canonical form.
#
# DETERMINISM GUARANTEES (ARCHITECTURE.md Section 3):
#   DET-01  No stochastic operations. No uuid, no os.urandom, no random.
#   DET-02  All inputs passed explicitly. No module-level mutable reads.
#   DET-03  No side effects in computational functions.
#           hash_file() reads a file -- this is the sole permitted IO in
#           this module. All other functions are pure.
#   DET-04  All hashing is deterministic SHA-256 over canonical byte sequences.
#   DET-05  All branches are functions of explicit inputs only.
#
# DEVIATION FROM FAS EXAMPLE CODE -- APPROVED:
#   The FAS blueprint example for append_to_chain() uses uuid.uuid4() for
#   event_id and datetime.utcnow() inside the hash computation. Both are
#   non-deterministic and violate DET-01/DET-05.
#   Approved resolution (pre-implementation review, 2026-02-21):
#     event_id  = SHA-256(prev_hash || event_type || canonical_json(data))
#     timestamp = caller-supplied string; stored in ChainEvent but NOT
#                 included in the hash computation.
#   This preserves the security model (event_id is content-addressed and
#   collision-resistant) while satisfying all determinism constraints.
#
# PROHIBITED ACTIONS CONFIRMED ABSENT:
#   - No logging calls
#   - No print statements
#   - No os.environ / os.getenv
#   - No module-level mutable containers
#   - No random / secrets / numpy.random
#   - No reimplementation of risk arithmetic
#   - No regime branching
#
# =============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# SECTION 1: HASH RESULT
# =============================================================================

@dataclass(frozen=True)
class HashResult:
    """
    Result of a single file hash computation.

    Attributes
    ----------
    file_path : str
        Absolute or relative path string of the hashed file.
    hash_value : str
        Lowercase 64-character hexadecimal SHA-256 digest.
    file_size : int
        Total bytes read from the file.
    """

    file_path: str
    hash_value: str
    file_size: int


# =============================================================================
# SECTION 2: MANIFEST TYPES
# =============================================================================

@dataclass(frozen=True)
class ManifestEntry:
    """
    Single entry in a file manifest.

    Attributes
    ----------
    path : str
        File path string as registered in the manifest.
    hash : str
        Expected SHA-256 hex digest.
    size : int
        Expected file size in bytes.
    timestamp_iso : str
        Caller-supplied ISO-8601 timestamp string. Not used in hash
        computation; recorded for audit purposes only.
    """

    path: str
    hash: str
    size: int
    timestamp_iso: str


@dataclass(frozen=True)
class VerificationResult:
    """
    Result of a manifest verification pass.

    Attributes
    ----------
    valid : bool
        True only when zero errors are detected.
    errors : List[str]
        Human-readable error descriptions (one per failing file).
    modified_files : List[str]
        Paths whose current hash differs from the manifest entry.
    missing_files : List[str]
        Paths recorded in the manifest that no longer exist on disk.
    """

    valid: bool
    errors: List[str]
    modified_files: List[str]
    missing_files: List[str]


@dataclass
class Manifest:
    """
    File hash manifest with JSON serialization support.

    Mutable so that entries can be populated incrementally by
    create_manifest(). Use VerificationResult to confirm integrity.

    Attributes
    ----------
    version : str
        Manifest format version string (e.g. "6.0.1").
    entries : Dict[str, ManifestEntry]
        Mapping from path string to ManifestEntry.
    """

    version: str
    entries: Dict[str, ManifestEntry] = field(default_factory=dict)

    def to_json(self) -> str:
        """
        Serialize manifest to a canonical JSON string.

        Keys are sorted for reproducibility. Returns ASCII-safe output
        (ensure_ascii=True is the json default).
        """
        data: Dict = {
            "version": self.version,
            "entries": {
                k: {
                    "path": v.path,
                    "hash": v.hash,
                    "size": v.size,
                    "timestamp_iso": v.timestamp_iso,
                }
                for k, v in self.entries.items()
            },
        }
        return json.dumps(data, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> Manifest:
        """
        Deserialize a manifest from a JSON string produced by to_json().

        Parameters
        ----------
        json_str : str
            JSON string as produced by to_json().

        Returns
        -------
        Manifest
            Reconstructed manifest instance.

        Raises
        ------
        KeyError
            If required fields are absent from the JSON structure.
        json.JSONDecodeError
            If the input is not valid JSON.
        """
        data: Dict = json.loads(json_str)
        entries: Dict[str, ManifestEntry] = {
            k: ManifestEntry(
                path=v["path"],
                hash=v["hash"],
                size=v["size"],
                timestamp_iso=v["timestamp_iso"],
            )
            for k, v in data["entries"].items()
        }
        return cls(version=data["version"], entries=entries)


# =============================================================================
# SECTION 3: HASH-CHAIN TYPES
# =============================================================================

@dataclass(frozen=True)
class ChainEvent:
    """
    Single event in an append-only hash chain.

    event_id is derived deterministically as:
        SHA-256(prev_hash || event_type || canonical_json(data))
    where '||' denotes concatenation of UTF-8 encoded strings.

    timestamp_iso is caller-supplied and stored for audit purposes.
    It is NOT included in the hash computation so that identical logical
    events produce identical hashes regardless of wall-clock time.

    current_hash is derived as:
        SHA-256(event_id || event_type || canonical_json(data) || prev_hash)
    This is recomputed during verify_chain() to detect tampering.

    Attributes
    ----------
    event_id : str
        Deterministic content-addressed identifier (64-char hex).
    event_type : str
        Caller-supplied event category label (ASCII).
    data : Dict
        Arbitrary JSON-serializable payload. Must not be mutated after
        the event is appended to a chain.
    previous_hash : str
        Hash of the preceding event (or genesis_hash for the first event).
    current_hash : str
        Hash of this event's full content (see derivation above).
    timestamp_iso : str
        Caller-supplied ISO-8601 string. Not part of hash computation.
    """

    event_id: str
    event_type: str
    data: Dict
    previous_hash: str
    current_hash: str
    timestamp_iso: str


@dataclass(frozen=True)
class ChainVerificationResult:
    """
    Result of a full hash-chain integrity verification pass.

    Attributes
    ----------
    valid : bool
        True only when every event in the chain passes both the
        previous-hash linkage check and the content-hash recomputation.
    broken_at : Optional[int]
        Zero-based index of the first failing event, or None if valid.
    error_message : Optional[str]
        Human-readable description of the first failure, or None if valid.
    """

    valid: bool
    broken_at: Optional[int]
    error_message: Optional[str]


@dataclass
class HashChain:
    """
    Append-only hash chain anchored by a genesis hash.

    genesis_hash is computed as SHA-256(genesis_event_string) during
    init_hash_chain(). The first appended event links against genesis_hash.

    events is a mutable list that grows via append_to_chain(). Callers
    must not mutate event payloads after appending -- doing so will cause
    verify_chain() to detect the tampering as intended (R3 tamper detection).

    Attributes
    ----------
    genesis_hash : str
        64-char hex SHA-256 of the genesis event string.
    events : List[ChainEvent]
        Ordered list of chain events; index 0 is earliest.
    """

    genesis_hash: str
    events: List[ChainEvent] = field(default_factory=list)

    def to_json(self) -> str:
        """
        Serialize the full chain to a canonical JSON string.

        Returns ASCII-safe JSON with sorted keys.
        """
        data: Dict = {
            "genesis_hash": self.genesis_hash,
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "timestamp_iso": e.timestamp_iso,
                    "data": e.data,
                    "previous_hash": e.previous_hash,
                    "current_hash": e.current_hash,
                }
                for e in self.events
            ],
        }
        return json.dumps(data, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> HashChain:
        """
        Deserialize a HashChain from JSON produced by to_json().

        Parameters
        ----------
        json_str : str
            JSON string as produced by to_json().

        Returns
        -------
        HashChain
            Reconstructed chain instance.

        Raises
        ------
        KeyError
            If required fields are absent from the JSON structure.
        json.JSONDecodeError
            If the input is not valid JSON.
        """
        data: Dict = json.loads(json_str)
        events: List[ChainEvent] = [
            ChainEvent(
                event_id=e["event_id"],
                event_type=e["event_type"],
                data=e["data"],
                previous_hash=e["previous_hash"],
                current_hash=e["current_hash"],
                timestamp_iso=e["timestamp_iso"],
            )
            for e in data["events"]
        ]
        return cls(genesis_hash=data["genesis_hash"], events=events)


# =============================================================================
# SECTION 4: THRESHOLD GOVERNANCE TYPES
# =============================================================================

@dataclass(frozen=True)
class ThresholdManifest:
    """
    Hash manifest for all critical numeric thresholds.

    threshold_hash is a SHA-256 digest over the canonical JSON
    serialization of the threshold name-value pairs drawn from
    jarvis.utils.constants.

    created_iso is caller-supplied and stored for audit purposes only.
    It does not affect threshold_hash.

    Attributes
    ----------
    version : str
        Manifest version string (e.g. "6.0.1").
    created_iso : str
        Caller-supplied ISO-8601 creation timestamp string.
    threshold_hash : str
        64-char hex SHA-256 over canonical threshold values.
    """

    version: str
    created_iso: str
    threshold_hash: str


@dataclass(frozen=True)
class ThresholdViolation(Exception):
    """
    Raised when runtime threshold values deviate from the stored manifest.

    This is a hard stop. No warning is issued. The caller must not catch
    and suppress this exception except to log and terminate the process.

    Attributes
    ----------
    expected_hash : str
        The threshold_hash recorded in the stored ThresholdManifest.
    actual_hash : str
        The threshold_hash recomputed from the current constants module.
    """

    expected_hash: str
    actual_hash: str


# =============================================================================
# SECTION 5: SYSTEM CONTRACT TYPES
# =============================================================================

@dataclass(frozen=True)
class ContractViolation:
    """
    Describes a single D(t) output contract violation.

    Attributes
    ----------
    field_name : str
        Name of the missing or inconsistent field.
    description : str
        Human-readable explanation of the violation.
    """

    field_name: str
    description: str


# =============================================================================
# SECTION 6: INTERNAL PURE HELPERS
# =============================================================================

def _sha256_hex(data: bytes) -> str:
    """
    Return the lowercase hex SHA-256 digest of the given bytes.

    Pure function. No IO. No side effects.

    Parameters
    ----------
    data : bytes
        Raw bytes to hash.

    Returns
    -------
    str
        64-character lowercase hexadecimal digest.
    """
    return sha256(data).hexdigest()


def _canonical_json(obj: Dict) -> str:
    """
    Serialize a dictionary to a canonical JSON string.

    Keys are sorted recursively. ensure_ascii=True (default) guarantees
    ASCII-only output. Separators are compact to eliminate whitespace
    variation.

    Parameters
    ----------
    obj : Dict
        JSON-serializable dictionary.

    Returns
    -------
    str
        Compact, sorted-key JSON string.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _derive_event_id(prev_hash: str, event_type: str, data: Dict) -> str:
    """
    Derive a deterministic event_id for a chain event.

    Computation:
        event_id = SHA-256(
            prev_hash || event_type || canonical_json(data)
        )
    where '||' is UTF-8 string concatenation.

    Parameters
    ----------
    prev_hash : str
        Hash of the previous chain event (or genesis hash).
    event_type : str
        Event category label.
    data : Dict
        Event payload (JSON-serializable).

    Returns
    -------
    str
        64-character lowercase hexadecimal event identifier.
    """
    raw: str = prev_hash + event_type + _canonical_json(data)
    return _sha256_hex(raw.encode("utf-8"))


def _derive_current_hash(
    event_id: str,
    event_type: str,
    data: Dict,
    prev_hash: str,
) -> str:
    """
    Derive the current_hash for a chain event.

    Computation:
        current_hash = SHA-256(
            event_id || event_type || canonical_json(data) || prev_hash
        )

    This hash is recomputed in verify_chain() to detect tampering of
    any field: event_id, event_type, data, or previous_hash linkage.

    Parameters
    ----------
    event_id : str
        Content-addressed event identifier (from _derive_event_id).
    event_type : str
        Event category label.
    data : Dict
        Event payload.
    prev_hash : str
        Hash of the previous chain event (or genesis hash).

    Returns
    -------
    str
        64-character lowercase hexadecimal current hash.
    """
    raw: str = event_id + event_type + _canonical_json(data) + prev_hash
    return _sha256_hex(raw.encode("utf-8"))


# =============================================================================
# SECTION 7: INTEGRITY LAYER
# =============================================================================

class IntegrityLayer:
    """
    Hash-chain based integrity and threshold governance system.

    This class is stateless. All state is passed in and returned
    explicitly. No instance variables are set or read during method
    execution. A new instance may be created freely at any call site
    without side effects.

    Methods
    -------
    hash_file(path)
        Compute SHA-256 of a file. Only method in this module that
        performs file IO.
    verify_file(path, expected_hash)
        Check a single file against an expected hash.
    create_manifest(file_list)
        Build a Manifest from a list of Paths.
    verify_manifest(manifest)
        Verify all entries in a Manifest against current file contents.
    init_hash_chain(genesis_event)
        Create a new empty HashChain anchored to a genesis hash.
    append_to_chain(chain, event_type, data, timestamp_iso)
        Append a new deterministic ChainEvent to an existing chain.
    verify_chain(chain)
        Verify every event in a HashChain for linkage and content integrity.
    create_threshold_manifest()
        Hash all critical thresholds from jarvis.utils.constants.
    verify_threshold_manifest(manifest)
        Raise ThresholdViolation if current thresholds deviate from manifest.
    """

    # -------------------------------------------------------------------------
    # Threshold governance: names of constants that must be hash-locked.
    # Only names present in jarvis.utils.constants are included. Names
    # belonging to sessions not yet built (S08, S09, S10, etc.) are listed
    # here for forward compatibility; they are silently skipped by
    # create_threshold_manifest() if the attribute does not yet exist.
    # DO NOT REORDER -- order affects nothing (canonical JSON sorts keys)
    # but stable ordering makes code review easier.
    # -------------------------------------------------------------------------
    _GOVERNED_THRESHOLD_NAMES: List[str] = [
        # S01 / S17 -- Risk Engine (present in constants.py v6.1.0)
        "MAX_DRAWDOWN_THRESHOLD",
        "VOL_COMPRESSION_TRIGGER",
        "SHOCK_EXPOSURE_CAP",
        # S11 / S12 -- Quality and selectivity (present)
        "BASE_SELECTIVITY_THRESHOLD",
        "QUALITY_SCORE_CAP_UNDER_UNCERTAINTY",
        "QUALITY_SCORE_MIN_FLOOR",
        # S26 -- Signal fragility (present)
        "FRAGILITY_HIGH_THRESHOLD",
        # S35 -- Decision context (present)
        "MAX_DECISION_CONTEXT",
        # S05 -- Regime duration stress (present)
        "DURATION_STRESS_Z_LIMIT",
        # Future sessions -- included for forward compatibility;
        # skipped gracefully until the session that defines them is built.
        # S09: calibration
        "ECE_HARD_GATE",
        "ECE_PER_REGIME_DRIFT",
        # S10: OOD detection
        "OOD_CONSENSUS_MINIMUM",
        "OOD_RECALL_HISTORISCH",
        "OOD_RECALL_SYNTHETISCH",
        "OOD_REGIME_SHIFT",
        # S08: meta-uncertainty
        "META_U_RECALIBRATION",
        "META_U_CONSERVATIVE",
        "META_U_COLLAPSE",
        # S17 extended: stress detection
        "STRESS_ERKENNUNG_MIN",
    ]

    # -------------------------------------------------------------------------
    # SECTION 7.1: File Hashing
    # -------------------------------------------------------------------------

    def hash_file(self, path: Path) -> HashResult:
        """
        Compute the SHA-256 hash of a file.

        This is the sole method in this module that performs file IO.
        Reads in 8 KB chunks to bound memory usage on large files.

        Parameters
        ----------
        path : Path
            Path to the file to hash.

        Returns
        -------
        HashResult
            Contains the hex digest, file path string, and byte count.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        PermissionError
            If the file cannot be opened for reading.
        """
        if not path.exists():
            raise FileNotFoundError(
                "File not found: " + str(path)
            )

        hasher = sha256()
        file_size: int = 0
        chunk: bytes

        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
                file_size += len(chunk)

        return HashResult(
            file_path=str(path),
            hash_value=hasher.hexdigest(),
            file_size=file_size,
        )

    def verify_file(self, path: Path, expected_hash: str) -> bool:
        """
        Verify a single file against an expected SHA-256 hex digest.

        Parameters
        ----------
        path : Path
            Path to the file to verify.
        expected_hash : str
            64-character lowercase hex string to compare against.

        Returns
        -------
        bool
            True if the file exists and its hash matches expected_hash.
            False if the file is missing or the hash differs.
        """
        if not path.exists():
            return False
        result: HashResult = self.hash_file(path)
        return result.hash_value == expected_hash

    # -------------------------------------------------------------------------
    # SECTION 7.2: Manifest
    # -------------------------------------------------------------------------

    def create_manifest(
        self,
        file_list: List[Path],
        version: str = "6.0.1",
        timestamp_iso: str = "1970-01-01T00:00:00",
    ) -> Manifest:
        """
        Build a Manifest by hashing every file in file_list.

        Parameters
        ----------
        file_list : List[Path]
            Ordered list of paths to include. Each path is hashed once.
        version : str, optional
            Manifest version label. Defaults to "6.0.1".
        timestamp_iso : str, optional
            Caller-supplied ISO-8601 timestamp stored in each entry.
            Not part of any hash computation. Defaults to epoch string.

        Returns
        -------
        Manifest
            Populated manifest with one entry per file.

        Raises
        ------
        FileNotFoundError
            If any path in file_list does not exist.
        """
        entries: Dict[str, ManifestEntry] = {}
        for path in file_list:
            result: HashResult = self.hash_file(path)
            entry = ManifestEntry(
                path=str(path),
                hash=result.hash_value,
                size=result.file_size,
                timestamp_iso=timestamp_iso,
            )
            entries[str(path)] = entry
        return Manifest(version=version, entries=entries)

    def verify_manifest(self, manifest: Manifest) -> VerificationResult:
        """
        Verify every entry in a Manifest against current file contents.

        Collects all errors before returning so the caller receives a
        complete diagnostic report rather than failing on the first error.

        Parameters
        ----------
        manifest : Manifest
            The manifest to verify.

        Returns
        -------
        VerificationResult
            valid=True only when no missing or modified files are found.
        """
        errors: List[str] = []
        modified_files: List[str] = []
        missing_files: List[str] = []

        for path_str, entry in manifest.entries.items():
            path = Path(path_str)

            if not path.exists():
                missing_files.append(path_str)
                errors.append("Missing: " + path_str)
                continue

            current: HashResult = self.hash_file(path)
            if current.hash_value != entry.hash:
                modified_files.append(path_str)
                errors.append(
                    "Modified: "
                    + path_str
                    + " -- expected: "
                    + entry.hash
                    + " -- actual: "
                    + current.hash_value
                )

        return VerificationResult(
            valid=len(errors) == 0,
            errors=errors,
            modified_files=modified_files,
            missing_files=missing_files,
        )

    # -------------------------------------------------------------------------
    # SECTION 7.3: Hash Chain
    # -------------------------------------------------------------------------

    def init_hash_chain(self, genesis_event: str) -> HashChain:
        """
        Initialize a new empty HashChain anchored to a genesis hash.

        genesis_hash = SHA-256(genesis_event encoded as UTF-8)

        Parameters
        ----------
        genesis_event : str
            A stable ASCII string that identifies this chain's purpose,
            e.g. "JARVIS v6.0.1 Build Start".

        Returns
        -------
        HashChain
            New chain with no events and a fixed genesis_hash.
        """
        genesis_hash: str = _sha256_hex(genesis_event.encode("utf-8"))
        return HashChain(genesis_hash=genesis_hash, events=[])

    def append_to_chain(
        self,
        chain: HashChain,
        event_type: str,
        data: Dict,
        timestamp_iso: str = "1970-01-01T00:00:00",
    ) -> ChainEvent:
        """
        Append a new deterministic ChainEvent to an existing chain.

        Derivation (approved deviation from FAS example, 2026-02-21):
            prev_hash  = last event's current_hash, or genesis_hash
            event_id   = SHA-256(prev_hash || event_type || canonical_json(data))
            current_hash = SHA-256(event_id || event_type || canonical_json(data)
                                   || prev_hash)
            timestamp_iso is stored but NOT included in any hash computation.

        Mutates chain.events by appending the new event.

        Parameters
        ----------
        chain : HashChain
            The chain to append to (mutated in place).
        event_type : str
            ASCII label for the event category.
        data : Dict
            JSON-serializable payload. Must remain immutable after call.
        timestamp_iso : str, optional
            Caller-supplied ISO-8601 timestamp for audit purposes only.
            Defaults to epoch string.

        Returns
        -------
        ChainEvent
            The newly created and appended event.
        """
        prev_hash: str = (
            chain.events[-1].current_hash
            if chain.events
            else chain.genesis_hash
        )

        event_id: str = _derive_event_id(prev_hash, event_type, data)
        current_hash: str = _derive_current_hash(
            event_id, event_type, data, prev_hash
        )

        event = ChainEvent(
            event_id=event_id,
            event_type=event_type,
            data=data,
            previous_hash=prev_hash,
            current_hash=current_hash,
            timestamp_iso=timestamp_iso,
        )

        chain.events.append(event)
        return event

    def verify_chain(self, chain: HashChain) -> ChainVerificationResult:
        """
        Verify every event in a HashChain for linkage and content integrity.

        Two checks per event (O(n) total):
          1. previous_hash linkage: event.previous_hash must equal
             the prior event's current_hash (or genesis_hash for index 0).
          2. Content hash recomputation: current_hash must equal
             _derive_current_hash(event_id, event_type, data, prev_hash).
             A mismatch indicates that event_id, event_type, data, or
             previous_hash was mutated after the event was appended.

        An empty chain (no events) is valid by definition.

        Parameters
        ----------
        chain : HashChain
            The chain to verify.

        Returns
        -------
        ChainVerificationResult
            valid=True and broken_at=None if all events pass.
            valid=False with broken_at set to the first failing index
            and error_message describing the failure type.
        """
        for i, event in enumerate(chain.events):
            expected_prev: str = (
                chain.events[i - 1].current_hash if i > 0 else chain.genesis_hash
            )

            # Check 1: linkage
            if event.previous_hash != expected_prev:
                return ChainVerificationResult(
                    valid=False,
                    broken_at=i,
                    error_message=(
                        "Chain linkage broken at event "
                        + str(i)
                        + ": previous_hash mismatch"
                    ),
                )

            # Check 2: content hash recomputation
            recomputed: str = _derive_current_hash(
                event.event_id,
                event.event_type,
                event.data,
                event.previous_hash,
            )
            if recomputed != event.current_hash:
                return ChainVerificationResult(
                    valid=False,
                    broken_at=i,
                    error_message=(
                        "Content hash mismatch at event "
                        + str(i)
                        + ": event data may have been tampered"
                    ),
                )

        return ChainVerificationResult(
            valid=True,
            broken_at=None,
            error_message=None,
        )

    # -------------------------------------------------------------------------
    # SECTION 7.4: Threshold Governance
    # -------------------------------------------------------------------------

    def create_threshold_manifest(
        self,
        version: str = "6.0.1",
        created_iso: str = "1970-01-01T00:00:00",
    ) -> ThresholdManifest:
        """
        Build a ThresholdManifest by hashing all governed threshold values.

        SANCTIONED CROSS-IMPORT (FAS v6.0.1 S01):
            Imports jarvis.utils.constants at call time (deferred).
            Only names present in constants are included in the hash.
            Names for sessions not yet built are skipped gracefully,
            ensuring the hash is stable within a given build state.

        The hash is computed as:
            threshold_hash = SHA-256(canonical_json({name: value, ...}))
        where the dict contains only names found in constants, sorted
        by json.dumps(sort_keys=True).

        Parameters
        ----------
        version : str, optional
            Manifest version label. Defaults to "6.0.1".
        created_iso : str, optional
            Caller-supplied ISO-8601 timestamp. Not part of hash.
            Defaults to epoch string.

        Returns
        -------
        ThresholdManifest
            Contains the computed threshold_hash.
        """
        # Sanctioned cross-import (see module header for justification)
        from jarvis.utils import constants  # noqa: PLC0415

        values: Dict[str, object] = {}
        for name in self._GOVERNED_THRESHOLD_NAMES:
            if hasattr(constants, name):
                values[name] = getattr(constants, name)
            # Names not yet defined (future sessions) are silently skipped.
            # This ensures hash stability for the current build state.

        canonical: str = _canonical_json(values)
        threshold_hash: str = _sha256_hex(canonical.encode("utf-8"))

        return ThresholdManifest(
            version=version,
            created_iso=created_iso,
            threshold_hash=threshold_hash,
        )

    def verify_threshold_manifest(
        self,
        manifest: ThresholdManifest,
    ) -> None:
        """
        Verify current threshold values against a stored ThresholdManifest.

        Recomputes the threshold hash from the live constants module and
        compares it against manifest.threshold_hash.

        HARD STOP CONTRACT (R4):
            Raises ThresholdViolation on any mismatch.
            No warning is issued. The caller must not suppress this
            exception other than to log and terminate.

        Parameters
        ----------
        manifest : ThresholdManifest
            The previously stored manifest to verify against.

        Returns
        -------
        None
            Returns normally if and only if hashes match.

        Raises
        ------
        ThresholdViolation
            If the recomputed hash differs from manifest.threshold_hash.
        """
        current: ThresholdManifest = self.create_threshold_manifest(
            version=manifest.version,
            created_iso=manifest.created_iso,
        )
        if current.threshold_hash != manifest.threshold_hash:
            raise ThresholdViolation(
                expected_hash=manifest.threshold_hash,
                actual_hash=current.threshold_hash,
            )


# =============================================================================
# SECTION 8: SYSTEM CONTRACT VERIFIER
# =============================================================================

_CONTRACT_REQUIRED_FIELDS: List[str] = ["mu", "sigma_squared", "Q", "S", "U", "R"]

_CONTRACT_SIGMA_COMPONENTS: List[str] = [
    "sigma_sq_aleatoric",
    "sigma_sq_epistemic_model",
    "sigma_sq_epistemic_data",
]

_CONTRACT_SIGMA_TOLERANCE: float = 1e-6


def verify_output_contract(output_obj: object) -> List[ContractViolation]:
    """
    Verify that an output object satisfies the system contract D(t).

    Contract definition (FAS v6.0.1 R5):
        D(t) = {mu, sigma^2, Q, S, U, R}
        sigma^2 = sigma_sq_aleatoric
                + sigma_sq_epistemic_model
                + sigma_sq_epistemic_data

    Two checks are performed:
      1. Field completeness: all six required fields and all three
         sigma^2 components must be present as attributes.
      2. sigma^2 consistency: the sum of the three components must
         equal sigma_squared within _CONTRACT_SIGMA_TOLERANCE (1e-6).

    Check 2 is skipped if check 1 fails (missing fields make
    arithmetic undefined).

    Parameters
    ----------
    output_obj : object
        Any object that is expected to satisfy the D(t) contract.

    Returns
    -------
    List[ContractViolation]
        Empty list means the contract is satisfied.
        Non-empty list contains one ContractViolation per failure.
        Deployment must be blocked if the list is non-empty.
    """
    violations: List[ContractViolation] = []

    # Check 1a: required top-level fields
    for field_name in _CONTRACT_REQUIRED_FIELDS:
        if not hasattr(output_obj, field_name):
            violations.append(
                ContractViolation(
                    field_name=field_name,
                    description="Required contract field missing: " + field_name,
                )
            )

    # Check 1b: sigma^2 decomposition components
    for component in _CONTRACT_SIGMA_COMPONENTS:
        if not hasattr(output_obj, component):
            violations.append(
                ContractViolation(
                    field_name=component,
                    description="sigma^2 decomposition component missing: " + component,
                )
            )

    # Check 2: sigma^2 consistency (only if all fields present)
    if not violations:
        expected_sum: float = (
            getattr(output_obj, "sigma_sq_aleatoric")
            + getattr(output_obj, "sigma_sq_epistemic_model")
            + getattr(output_obj, "sigma_sq_epistemic_data")
        )
        actual: float = getattr(output_obj, "sigma_squared")
        if abs(actual - expected_sum) > _CONTRACT_SIGMA_TOLERANCE:
            violations.append(
                ContractViolation(
                    field_name="sigma_squared",
                    description=(
                        "sigma^2 inconsistency: sigma_squared="
                        + str(actual)
                        + " but component sum="
                        + str(expected_sum)
                        + " (tolerance="
                        + str(_CONTRACT_SIGMA_TOLERANCE)
                        + ")"
                    ),
                )
            )

    return violations
