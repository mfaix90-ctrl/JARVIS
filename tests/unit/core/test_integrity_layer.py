# tests/unit/core/test_integrity_layer.py
# MASP v1.2.0-G — STRICT MODE
# Target: jarvis/core/integrity_layer.py
# No mocks, no refactors, no side effects, deterministic.

from __future__ import annotations

import json
import tempfile
from hashlib import sha256
from pathlib import Path

import pytest

from jarvis.core.integrity_layer import (
    ChainEvent,
    ChainVerificationResult,
    ContractViolation,
    HashChain,
    HashResult,
    IntegrityLayer,
    Manifest,
    ManifestEntry,
    ThresholdManifest,
    ThresholdViolation,
    VerificationResult,
    _canonical_json,
    _derive_current_hash,
    _derive_event_id,
    _sha256_hex,
    verify_output_contract,
)


# =============================================================================
# Helpers
# =============================================================================

def _make_layer() -> IntegrityLayer:
    return IntegrityLayer()


def _write_temp_file(content: bytes) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _known_sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


# =============================================================================
# SECTION 6: Internal Pure Helpers
# =============================================================================

class TestSha256Hex:
    def test_empty_bytes(self):
        result = _sha256_hex(b"")
        assert result == sha256(b"").hexdigest()

    def test_known_value(self):
        data = b"hello"
        assert _sha256_hex(data) == sha256(b"hello").hexdigest()

    def test_returns_64_char_lowercase_hex(self):
        result = _sha256_hex(b"abc")
        assert len(result) == 64
        assert result == result.lower()
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic_same_input(self):
        assert _sha256_hex(b"test") == _sha256_hex(b"test")

    def test_different_inputs_differ(self):
        assert _sha256_hex(b"a") != _sha256_hex(b"b")


class TestCanonicalJson:
    def test_empty_dict(self):
        result = _canonical_json({})
        assert result == "{}"

    def test_keys_sorted(self):
        result = _canonical_json({"z": 1, "a": 2})
        parsed = json.loads(result)
        assert list(json.loads(result).keys()) == sorted(parsed.keys())
        assert '"a":2' in result
        assert '"z":1' in result

    def test_compact_no_spaces(self):
        result = _canonical_json({"k": "v"})
        assert " " not in result

    def test_deterministic(self):
        obj = {"b": 2, "a": 1}
        assert _canonical_json(obj) == _canonical_json(obj)

    def test_nested_dict_serialized(self):
        result = _canonical_json({"outer": {"inner": 1}})
        assert "inner" in result

    def test_numeric_values(self):
        result = _canonical_json({"x": 3.14})
        assert "3.14" in result


class TestDeriveEventId:
    def test_returns_64_char_hex(self):
        eid = _derive_event_id("prev", "TYPE", {"k": "v"})
        assert len(eid) == 64
        assert eid == eid.lower()

    def test_deterministic(self):
        a = _derive_event_id("prev", "TYPE", {"k": "v"})
        b = _derive_event_id("prev", "TYPE", {"k": "v"})
        assert a == b

    def test_different_prev_hash_differs(self):
        a = _derive_event_id("aaa", "TYPE", {"k": "v"})
        b = _derive_event_id("bbb", "TYPE", {"k": "v"})
        assert a != b

    def test_different_event_type_differs(self):
        a = _derive_event_id("prev", "A", {})
        b = _derive_event_id("prev", "B", {})
        assert a != b

    def test_different_data_differs(self):
        a = _derive_event_id("prev", "T", {"x": 1})
        b = _derive_event_id("prev", "T", {"x": 2})
        assert a != b

    def test_known_computation(self):
        prev_hash = "prev"
        event_type = "T"
        data = {"k": "v"}
        canonical = _canonical_json(data)
        raw = (prev_hash + event_type + canonical).encode("utf-8")
        expected = sha256(raw).hexdigest()
        assert _derive_event_id(prev_hash, event_type, data) == expected


class TestDeriveCurrentHash:
    def test_returns_64_char_hex(self):
        ch = _derive_current_hash("eid", "TYPE", {"k": "v"}, "prev")
        assert len(ch) == 64
        assert ch == ch.lower()

    def test_deterministic(self):
        a = _derive_current_hash("eid", "T", {}, "p")
        b = _derive_current_hash("eid", "T", {}, "p")
        assert a == b

    def test_different_event_id_differs(self):
        a = _derive_current_hash("eid1", "T", {}, "p")
        b = _derive_current_hash("eid2", "T", {}, "p")
        assert a != b

    def test_different_prev_hash_differs(self):
        a = _derive_current_hash("eid", "T", {}, "p1")
        b = _derive_current_hash("eid", "T", {}, "p2")
        assert a != b

    def test_known_computation(self):
        event_id = "eid"
        event_type = "T"
        data = {"k": "v"}
        prev_hash = "prev"
        canonical = _canonical_json(data)
        raw = (event_id + event_type + canonical + prev_hash).encode("utf-8")
        expected = sha256(raw).hexdigest()
        assert _derive_current_hash(event_id, event_type, data, prev_hash) == expected


# =============================================================================
# SECTION 1: HashResult
# =============================================================================

class TestHashResult:
    def test_is_frozen(self):
        hr = HashResult(file_path="/a", hash_value="abc", file_size=0)
        with pytest.raises(Exception):
            hr.file_path = "/b"  # type: ignore[misc]

    def test_fields_stored(self):
        hr = HashResult(file_path="/x", hash_value="deadbeef", file_size=42)
        assert hr.file_path == "/x"
        assert hr.hash_value == "deadbeef"
        assert hr.file_size == 42


# =============================================================================
# SECTION 2: Manifest and ManifestEntry
# =============================================================================

class TestManifestEntry:
    def test_is_frozen(self):
        me = ManifestEntry(path="p", hash="h", size=0, timestamp_iso="T")
        with pytest.raises(Exception):
            me.path = "q"  # type: ignore[misc]


class TestManifest:
    def _sample(self) -> Manifest:
        entries = {
            "file.py": ManifestEntry(
                path="file.py",
                hash="a" * 64,
                size=100,
                timestamp_iso="2000-01-01T00:00:00",
            )
        }
        return Manifest(version="6.0.1", entries=entries)

    def test_to_json_is_valid_json(self):
        m = self._sample()
        j = m.to_json()
        parsed = json.loads(j)
        assert parsed["version"] == "6.0.1"

    def test_to_json_has_entries(self):
        m = self._sample()
        parsed = json.loads(m.to_json())
        assert "file.py" in parsed["entries"]

    def test_roundtrip(self):
        m = self._sample()
        m2 = Manifest.from_json(m.to_json())
        assert m2.version == m.version
        assert set(m2.entries.keys()) == set(m.entries.keys())
        e1 = m.entries["file.py"]
        e2 = m2.entries["file.py"]
        assert e1.hash == e2.hash
        assert e1.size == e2.size
        assert e1.timestamp_iso == e2.timestamp_iso

    def test_from_json_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            Manifest.from_json("not json")

    def test_from_json_missing_version_key_raises(self):
        broken = json.dumps({"entries": {}})
        with pytest.raises(KeyError):
            Manifest.from_json(broken)

    def test_from_json_missing_entries_key_raises(self):
        broken = json.dumps({"version": "1"})
        with pytest.raises(KeyError):
            Manifest.from_json(broken)

    def test_empty_manifest_roundtrip(self):
        m = Manifest(version="1.0", entries={})
        m2 = Manifest.from_json(m.to_json())
        assert m2.version == "1.0"
        assert m2.entries == {}

    def test_to_json_keys_sorted(self):
        m = self._sample()
        raw = m.to_json()
        idx_entries = raw.find('"entries"')
        idx_version = raw.find('"version"')
        assert idx_entries < idx_version


# =============================================================================
# SECTION 3: HashChain, ChainEvent, ChainVerificationResult
# =============================================================================

class TestHashChain:
    def test_to_json_from_json_roundtrip_empty(self):
        chain = HashChain(genesis_hash="g" * 64, events=[])
        chain2 = HashChain.from_json(chain.to_json())
        assert chain2.genesis_hash == chain.genesis_hash
        assert chain2.events == []

    def test_from_json_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            HashChain.from_json("bad json")

    def test_from_json_missing_genesis_hash_raises(self):
        broken = json.dumps({"events": []})
        with pytest.raises(KeyError):
            HashChain.from_json(broken)

    def test_roundtrip_with_events(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        layer.append_to_chain(chain, "T", {"x": 1})
        layer.append_to_chain(chain, "U", {"y": 2})
        restored = HashChain.from_json(chain.to_json())
        assert restored.genesis_hash == chain.genesis_hash
        assert len(restored.events) == 2
        assert restored.events[0].event_id == chain.events[0].event_id
        assert restored.events[1].current_hash == chain.events[1].current_hash


class TestChainVerificationResult:
    def test_is_frozen(self):
        r = ChainVerificationResult(valid=True, broken_at=None, error_message=None)
        with pytest.raises(Exception):
            r.valid = False  # type: ignore[misc]


# =============================================================================
# SECTION 7.1: IntegrityLayer.hash_file / verify_file
# =============================================================================

class TestHashFile:
    def test_empty_file(self):
        path = _write_temp_file(b"")
        layer = _make_layer()
        result = layer.hash_file(path)
        assert result.hash_value == _known_sha256(b"")
        assert result.file_size == 0
        assert result.file_path == str(path)

    def test_known_content(self):
        content = b"hello world"
        path = _write_temp_file(content)
        layer = _make_layer()
        result = layer.hash_file(path)
        assert result.hash_value == _known_sha256(content)
        assert result.file_size == len(content)

    def test_file_not_found_raises(self):
        layer = _make_layer()
        path = Path("/nonexistent/path/file.bin")
        with pytest.raises(FileNotFoundError):
            layer.hash_file(path)

    def test_deterministic_same_file(self):
        path = _write_temp_file(b"deterministic")
        layer = _make_layer()
        r1 = layer.hash_file(path)
        r2 = layer.hash_file(path)
        assert r1.hash_value == r2.hash_value

    def test_large_content(self):
        content = b"x" * (8192 * 3 + 7)  # spans multiple 8 KB chunks
        path = _write_temp_file(content)
        layer = _make_layer()
        result = layer.hash_file(path)
        assert result.hash_value == _known_sha256(content)
        assert result.file_size == len(content)

    def test_returns_hash_result_type(self):
        path = _write_temp_file(b"t")
        result = _make_layer().hash_file(path)
        assert isinstance(result, HashResult)


class TestVerifyFile:
    def test_correct_hash_returns_true(self):
        content = b"verify me"
        path = _write_temp_file(content)
        expected = _known_sha256(content)
        assert _make_layer().verify_file(path, expected) is True

    def test_wrong_hash_returns_false(self):
        content = b"verify me"
        path = _write_temp_file(content)
        assert _make_layer().verify_file(path, "0" * 64) is False

    def test_missing_file_returns_false(self):
        path = Path("/no/such/file.txt")
        assert _make_layer().verify_file(path, "a" * 64) is False


# =============================================================================
# SECTION 7.2: create_manifest / verify_manifest
# =============================================================================

class TestCreateManifest:
    def test_single_file(self):
        content = b"manifest test"
        path = _write_temp_file(content)
        layer = _make_layer()
        m = layer.create_manifest([path])
        assert str(path) in m.entries
        entry = m.entries[str(path)]
        assert entry.hash == _known_sha256(content)
        assert entry.size == len(content)

    def test_version_stored(self):
        path = _write_temp_file(b"v")
        m = _make_layer().create_manifest([path], version="9.9.9")
        assert m.version == "9.9.9"

    def test_timestamp_stored_not_in_hash(self):
        path = _write_temp_file(b"ts")
        m = _make_layer().create_manifest([path], timestamp_iso="2099-01-01T00:00:00")
        entry = m.entries[str(path)]
        assert entry.timestamp_iso == "2099-01-01T00:00:00"

    def test_multiple_files(self):
        paths = [_write_temp_file(b"file" + bytes([i])) for i in range(3)]
        m = _make_layer().create_manifest(paths)
        for p in paths:
            assert str(p) in m.entries

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            _make_layer().create_manifest([Path("/no/such/file.bin")])

    def test_empty_file_list(self):
        m = _make_layer().create_manifest([])
        assert m.entries == {}


class TestVerifyManifest:
    def test_valid_manifest(self):
        content = b"valid"
        path = _write_temp_file(content)
        layer = _make_layer()
        m = layer.create_manifest([path])
        result = layer.verify_manifest(m)
        assert result.valid is True
        assert result.errors == []
        assert result.modified_files == []
        assert result.missing_files == []

    def test_missing_file_detected(self):
        entry = ManifestEntry(
            path="/nonexistent/ghost.py",
            hash="a" * 64,
            size=0,
            timestamp_iso="T",
        )
        m = Manifest(version="1.0", entries={"/nonexistent/ghost.py": entry})
        result = _make_layer().verify_manifest(m)
        assert result.valid is False
        assert "/nonexistent/ghost.py" in result.missing_files
        assert len(result.errors) == 1

    def test_modified_file_detected(self):
        path = _write_temp_file(b"original")
        layer = _make_layer()
        m = layer.create_manifest([path])
        # Overwrite with different content
        path.write_bytes(b"tampered")
        result = layer.verify_manifest(m)
        assert result.valid is False
        assert str(path) in result.modified_files

    def test_multiple_errors_all_reported(self):
        ghost = ManifestEntry(path="/no/a.py", hash="a" * 64, size=0, timestamp_iso="T")
        ghost2 = ManifestEntry(path="/no/b.py", hash="b" * 64, size=0, timestamp_iso="T")
        m = Manifest(version="1.0", entries={"/no/a.py": ghost, "/no/b.py": ghost2})
        result = _make_layer().verify_manifest(m)
        assert result.valid is False
        assert len(result.errors) == 2
        assert len(result.missing_files) == 2

    def test_empty_manifest_is_valid(self):
        m = Manifest(version="1.0", entries={})
        result = _make_layer().verify_manifest(m)
        assert result.valid is True

    def test_returns_verification_result_type(self):
        m = Manifest(version="1.0", entries={})
        result = _make_layer().verify_manifest(m)
        assert isinstance(result, VerificationResult)


# =============================================================================
# SECTION 7.3: init_hash_chain / append_to_chain / verify_chain
# =============================================================================

class TestInitHashChain:
    def test_genesis_hash_is_sha256_of_string(self):
        layer = _make_layer()
        genesis_event = "JARVIS v6.0.1 Build Start"
        chain = layer.init_hash_chain(genesis_event)
        expected = sha256(genesis_event.encode("utf-8")).hexdigest()
        assert chain.genesis_hash == expected

    def test_empty_events(self):
        chain = _make_layer().init_hash_chain("genesis")
        assert chain.events == []

    def test_deterministic(self):
        a = _make_layer().init_hash_chain("test")
        b = _make_layer().init_hash_chain("test")
        assert a.genesis_hash == b.genesis_hash

    def test_different_genesis_strings_differ(self):
        a = _make_layer().init_hash_chain("A")
        b = _make_layer().init_hash_chain("B")
        assert a.genesis_hash != b.genesis_hash


class TestAppendToChain:
    def test_first_event_links_to_genesis(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        event = layer.append_to_chain(chain, "T", {"k": "v"})
        assert event.previous_hash == chain.genesis_hash

    def test_second_event_links_to_first(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        e1 = layer.append_to_chain(chain, "T", {"i": 1})
        e2 = layer.append_to_chain(chain, "U", {"i": 2})
        assert e2.previous_hash == e1.current_hash

    def test_mutates_chain_events(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        assert len(chain.events) == 0
        layer.append_to_chain(chain, "T", {})
        assert len(chain.events) == 1
        layer.append_to_chain(chain, "U", {})
        assert len(chain.events) == 2

    def test_returns_chain_event_type(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        event = layer.append_to_chain(chain, "T", {})
        assert isinstance(event, ChainEvent)

    def test_event_id_is_64_char_hex(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        event = layer.append_to_chain(chain, "T", {})
        assert len(event.event_id) == 64
        assert event.event_id == event.event_id.lower()

    def test_timestamp_stored_not_affecting_hashes(self):
        layer = _make_layer()
        chain_a = layer.init_hash_chain("genesis")
        chain_b = layer.init_hash_chain("genesis")
        e_a = layer.append_to_chain(chain_a, "T", {"k": "v"}, timestamp_iso="T1")
        e_b = layer.append_to_chain(chain_b, "T", {"k": "v"}, timestamp_iso="T2")
        assert e_a.event_id == e_b.event_id
        assert e_a.current_hash == e_b.current_hash
        assert e_a.timestamp_iso != e_b.timestamp_iso

    def test_deterministic_same_inputs(self):
        layer = _make_layer()
        c1 = layer.init_hash_chain("genesis")
        c2 = layer.init_hash_chain("genesis")
        ev1 = layer.append_to_chain(c1, "T", {"x": 1})
        ev2 = layer.append_to_chain(c2, "T", {"x": 1})
        assert ev1.event_id == ev2.event_id
        assert ev1.current_hash == ev2.current_hash


class TestVerifyChain:
    def test_empty_chain_is_valid(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        result = layer.verify_chain(chain)
        assert result.valid is True
        assert result.broken_at is None
        assert result.error_message is None

    def test_valid_single_event(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        layer.append_to_chain(chain, "T", {"x": 1})
        result = layer.verify_chain(chain)
        assert result.valid is True

    def test_valid_multi_event(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        for i in range(5):
            layer.append_to_chain(chain, "T", {"i": i})
        result = layer.verify_chain(chain)
        assert result.valid is True

    def test_tampered_previous_hash_detected(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        layer.append_to_chain(chain, "T", {"k": "v"})
        # Replace event with tampered previous_hash using object.__setattr__
        original = chain.events[0]
        tampered = ChainEvent(
            event_id=original.event_id,
            event_type=original.event_type,
            data=original.data,
            previous_hash="tampered" + "0" * 57,
            current_hash=original.current_hash,
            timestamp_iso=original.timestamp_iso,
        )
        chain.events[0] = tampered
        result = layer.verify_chain(chain)
        assert result.valid is False
        assert result.broken_at == 0
        assert "linkage" in result.error_message.lower() or "previous_hash" in result.error_message.lower()

    def test_tampered_data_detected(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        layer.append_to_chain(chain, "T", {"k": "original"})
        original = chain.events[0]
        tampered = ChainEvent(
            event_id=original.event_id,
            event_type=original.event_type,
            data={"k": "tampered"},
            previous_hash=original.previous_hash,
            current_hash=original.current_hash,
            timestamp_iso=original.timestamp_iso,
        )
        chain.events[0] = tampered
        result = layer.verify_chain(chain)
        assert result.valid is False
        assert result.broken_at == 0
        assert "tamper" in result.error_message.lower() or "mismatch" in result.error_message.lower()

    def test_broken_at_index_correct_for_second_event(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        layer.append_to_chain(chain, "A", {"x": 1})
        layer.append_to_chain(chain, "B", {"x": 2})
        original = chain.events[1]
        tampered = ChainEvent(
            event_id=original.event_id,
            event_type=original.event_type,
            data={"x": 999},
            previous_hash=original.previous_hash,
            current_hash=original.current_hash,
            timestamp_iso=original.timestamp_iso,
        )
        chain.events[1] = tampered
        result = layer.verify_chain(chain)
        assert result.valid is False
        assert result.broken_at == 1

    def test_returns_chain_verification_result_type(self):
        layer = _make_layer()
        chain = layer.init_hash_chain("genesis")
        assert isinstance(layer.verify_chain(chain), ChainVerificationResult)


# =============================================================================
# SECTION 7.4: create_threshold_manifest / verify_threshold_manifest
# =============================================================================

class TestCreateThresholdManifest:
    def test_returns_threshold_manifest(self):
        tm = _make_layer().create_threshold_manifest()
        assert isinstance(tm, ThresholdManifest)

    def test_hash_is_64_char_hex(self):
        tm = _make_layer().create_threshold_manifest()
        assert len(tm.threshold_hash) == 64
        assert tm.threshold_hash == tm.threshold_hash.lower()

    def test_version_stored(self):
        tm = _make_layer().create_threshold_manifest(version="TEST")
        assert tm.version == "TEST"

    def test_created_iso_stored(self):
        tm = _make_layer().create_threshold_manifest(created_iso="2099-12-31T00:00:00")
        assert tm.created_iso == "2099-12-31T00:00:00"

    def test_deterministic(self):
        a = _make_layer().create_threshold_manifest()
        b = _make_layer().create_threshold_manifest()
        assert a.threshold_hash == b.threshold_hash

    def test_created_iso_does_not_affect_hash(self):
        a = _make_layer().create_threshold_manifest(created_iso="T1")
        b = _make_layer().create_threshold_manifest(created_iso="T2")
        assert a.threshold_hash == b.threshold_hash


class TestVerifyThresholdManifest:
    def test_valid_manifest_does_not_raise(self):
        layer = _make_layer()
        tm = layer.create_threshold_manifest()
        layer.verify_threshold_manifest(tm)  # must not raise

    def test_tampered_hash_raises_threshold_violation(self):
        layer = _make_layer()
        tm = layer.create_threshold_manifest()
        tampered = ThresholdManifest(
            version=tm.version,
            created_iso=tm.created_iso,
            threshold_hash="0" * 64,
        )
        with pytest.raises(ThresholdViolation):
            layer.verify_threshold_manifest(tampered)

    def test_threshold_violation_contains_hashes(self):
        layer = _make_layer()
        tm = layer.create_threshold_manifest()
        tampered = ThresholdManifest(
            version=tm.version,
            created_iso=tm.created_iso,
            threshold_hash="f" * 64,
        )
        with pytest.raises(ThresholdViolation) as exc_info:
            layer.verify_threshold_manifest(tampered)
        violation = exc_info.value
        assert violation.expected_hash == "f" * 64
        assert violation.actual_hash == tm.threshold_hash


# =============================================================================
# SECTION 8: verify_output_contract
# =============================================================================

import types as _types

# Canonical full-contract field values.
# All nine fields present; sigma_squared == sum of three components exactly.
_FULL_FIELDS: dict = dict(
    mu=0.1,
    sigma_squared=0.3,
    Q=1.0,
    S=2.0,
    U=0.5,
    R=0.9,
    sigma_sq_aleatoric=0.1,
    sigma_sq_epistemic_model=0.1,
    sigma_sq_epistemic_data=0.1,
)


def _contract_obj(**overrides) -> object:
    """
    Build a SimpleNamespace with all nine contract fields.
    Fields listed in overrides replace canonical defaults.
    Pass a field with value _MISSING to exclude it entirely.
    """
    _MISSING = object()
    fields = {**_FULL_FIELDS, **overrides}
    return _types.SimpleNamespace(**{k: v for k, v in fields.items() if v is not _MISSING})


_MISSING_SENTINEL = object()  # stable sentinel for exclusion


def _contract_obj_without(*exclude: str, **overrides) -> object:
    """Return a SimpleNamespace with all full fields except those in *exclude*."""
    fields = {k: v for k, v in _FULL_FIELDS.items() if k not in exclude}
    fields.update(overrides)
    return _types.SimpleNamespace(**fields)


class TestVerifyOutputContract:
    def test_valid_object_no_violations(self):
        obj = _contract_obj_without()
        assert verify_output_contract(obj) == []

    def test_missing_mu_detected(self):
        obj = _contract_obj_without("mu")
        violations = verify_output_contract(obj)
        assert any(v.field_name == "mu" for v in violations)

    def test_missing_sigma_squared_detected(self):
        obj = _contract_obj_without("sigma_squared")
        violations = verify_output_contract(obj)
        assert any(v.field_name == "sigma_squared" for v in violations)

    def test_missing_Q_detected(self):
        obj = _contract_obj_without("Q")
        violations = verify_output_contract(obj)
        assert any(v.field_name == "Q" for v in violations)

    def test_missing_S_detected(self):
        obj = _contract_obj_without("S")
        violations = verify_output_contract(obj)
        assert any(v.field_name == "S" for v in violations)

    def test_missing_U_detected(self):
        obj = _contract_obj_without("U")
        violations = verify_output_contract(obj)
        assert any(v.field_name == "U" for v in violations)

    def test_missing_R_detected(self):
        obj = _contract_obj_without("R")
        violations = verify_output_contract(obj)
        assert any(v.field_name == "R" for v in violations)

    def test_missing_sigma_component_aleatoric(self):
        obj = _contract_obj_without("sigma_sq_aleatoric")
        violations = verify_output_contract(obj)
        assert any(v.field_name == "sigma_sq_aleatoric" for v in violations)

    def test_missing_sigma_component_model(self):
        obj = _contract_obj_without("sigma_sq_epistemic_model")
        violations = verify_output_contract(obj)
        assert any(v.field_name == "sigma_sq_epistemic_model" for v in violations)

    def test_missing_sigma_component_data(self):
        obj = _contract_obj_without("sigma_sq_epistemic_data")
        violations = verify_output_contract(obj)
        assert any(v.field_name == "sigma_sq_epistemic_data" for v in violations)

    def test_sigma_inconsistency_detected(self):
        # sigma_squared does not equal sum of components
        obj = _contract_obj_without(sigma_squared=999.0)
        violations = verify_output_contract(obj)
        assert any(v.field_name == "sigma_squared" for v in violations)

    def test_sigma_consistency_within_tolerance(self):
        # sum = 0.29999999, sigma_squared = 0.29999999 → diff < 1e-6
        obj = _contract_obj_without(
            sigma_sq_aleatoric=0.1,
            sigma_sq_epistemic_model=0.1,
            sigma_sq_epistemic_data=0.09999999,
            sigma_squared=0.29999999,
        )
        assert verify_output_contract(obj) == []

    def test_sigma_inconsistency_just_outside_tolerance(self):
        # diff = 2e-6 > 1e-6 → violation
        obj = _contract_obj_without(
            sigma_sq_aleatoric=0.1,
            sigma_sq_epistemic_model=0.1,
            sigma_sq_epistemic_data=0.1,
            sigma_squared=0.3 + 2e-6,
        )
        violations = verify_output_contract(obj)
        assert any(v.field_name == "sigma_squared" for v in violations)

    def test_check2_skipped_when_fields_missing(self):
        # No fields at all → only field-missing violations, no sigma inconsistency
        obj = _types.SimpleNamespace()
        violations = verify_output_contract(obj)
        sigma_violations = [v for v in violations if "inconsistency" in v.description]
        assert sigma_violations == []

    def test_empty_object_all_fields_missing(self):
        obj = _types.SimpleNamespace()
        violations = verify_output_contract(obj)
        field_names = {v.field_name for v in violations}
        for required in [
            "mu", "sigma_squared", "Q", "S", "U", "R",
            "sigma_sq_aleatoric", "sigma_sq_epistemic_model",
            "sigma_sq_epistemic_data",
        ]:
            assert required in field_names

    def test_returns_list(self):
        obj = _contract_obj_without()
        assert isinstance(verify_output_contract(obj), list)

    def test_violation_fields_populated(self):
        obj = _contract_obj_without("mu")
        violations = verify_output_contract(obj)
        v = next(v for v in violations if v.field_name == "mu")
        assert isinstance(v, ContractViolation)
        assert "mu" in v.description
        assert isinstance(v.description, str)
