# jarvis/verification/run_harness.py
# Deterministic Verification Harness -- Entry Point.
# Authority: DVH Implementation Blueprint v1.0.0 Section 6 and 18.
#
# Standard invocation (DIM-03):
#   python -m jarvis.verification.run_harness \
#       --manifest-path jarvis/risk/THRESHOLD_MANIFEST.json \
#       --module-version 6.1.0 \
#       --runs-dir jarvis/verification/runs
#
# Cross-session replay (DIM-05):
#   python -m jarvis.verification.run_harness \
#       --manifest-path jarvis/risk/THRESHOLD_MANIFEST.json \
#       --module-version 6.1.0 \
#       --runs-dir jarvis/verification/runs \
#       --cross-session-replay \
#       --prior-record-path [path_to_ER_record.json]
#
# EXIT CODES (HFP-01):
#   0  -- All verification passes passed. FREEZE compliant.
#   1  -- FREEZE_VIOLATION or DETERMINISM_BREACH.
#   2  -- INTEGRITY_FAILURE (manifest issues).
#   3  -- DATA_CORRUPTION or CONTRACT_VIOLATION.
#   4  -- Internal harness error.
#
# EEP-01: Single-threaded. No subprocesses. No async tasks.
# EEP-07: No random number generation.
# OC-01: 10-minute wall-clock timeout (600 seconds).

import argparse
import json
import sys
import uuid
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from jarvis.verification.harness_version import (
    HARNESS_VERSION,
    EXPECTED_MODULE_VERSION,
    STORAGE_FORMAT_VERSION,
)
from jarvis.verification.manifest_validator import ManifestValidator
from jarvis.verification.input_vector_generator import InputVectorGenerator
from jarvis.verification.execution_recorder import ExecutionRecorder
from jarvis.verification.replay_engine import ReplayEngine
from jarvis.verification.bit_comparator import BitComparator
from jarvis.verification.clip_verifier import ClipVerifier
from jarvis.verification.failure_handler import FailureHandler
from jarvis.verification.storage.record_serializer import RecordSerializer
from jarvis.verification.storage.record_loader import RecordLoader
from jarvis.verification.data_models.execution_record import ExecutionRecord
from jarvis.verification.data_models.comparison_report import ComparisonReport


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JARVIS Deterministic Verification Harness v1.0.0",
        prog="python -m jarvis.verification.run_harness",
    )
    parser.add_argument(
        "--manifest-path",
        required=True,
        help="Path to THRESHOLD_MANIFEST.json (MVI-01).",
    )
    parser.add_argument(
        "--module-version",
        required=True,
        help="Risk Engine module version to verify. Must equal '6.1.0' (DIM-04).",
    )
    parser.add_argument(
        "--runs-dir",
        required=True,
        help="Directory for output run records (RSF-05).",
    )
    parser.add_argument(
        "--cross-session-replay",
        action="store_true",
        default=False,
        help="Load prior ER records for cross-session replay (DIM-05/DIM-06).",
    )
    parser.add_argument(
        "--prior-record-path",
        default=None,
        help="Path to prior ER record JSON file. Required with --cross-session-replay.",
    )
    return parser.parse_args()


def _install_timeout(seconds: int, fh: Optional[FailureHandler]) -> None:
    """Install SIGALRM-based timeout (OC-01). Unix only."""
    try:
        def _timeout_handler(signum, frame):
            msg = (
                f"HARNESS_INTERNAL_ERROR: Wall-clock timeout of {seconds}s exceeded "
                f"(OC-01). Harness aborted."
            )
            if fh is not None:
                fh.handle("HARNESS_INTERNAL_ERROR", msg)
            else:
                sys.stderr.write(msg + "\n")
                sys.exit(4)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(seconds)
    except AttributeError:
        # Windows does not have SIGALRM. Timeout is advisory on Windows.
        pass


def main() -> None:
    """
    Main harness pipeline. Single-threaded. Deterministic. No side effects
    on the production Risk Engine module.

    Pipeline sequence (Section 3 of DVH Architecture):
      MV (ManifestValidator)
      IVG (InputVectorGenerator)
      ER (ExecutionRecorder)  or  RecordLoader in cross-session replay
      RE (ReplayEngine)
      BIC (BitComparator)
      CCV (ClipVerifier)
      FH (FailureHandler) -- invoked only on failure

    On pass: prints summary and exits 0.
    On any failure: FailureHandler invokes sys.exit(non-zero).
    """
    args       = _parse_args()
    run_id     = "RUN-" + datetime.now(timezone.utc).strftime("%Y%m%d") + "-" + str(uuid.uuid4())[:8].upper()
    runs_dir   = Path(args.runs_dir)
    manifest_path = Path(args.manifest_path)

    # Verify runs directory write access before pipeline begins (RSF-05).
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
        test_file = runs_dir / f".write_test_{run_id}"
        test_file.touch()
        test_file.unlink()
    except Exception as exc:
        sys.stderr.write(
            f"INTEGRITY_FAILURE: Cannot write to runs directory {runs_dir}: {exc}\n"
        )
        sys.exit(2)

    # Initialise FailureHandler early for timeout handler.
    fh = FailureHandler(
        runs_dir=runs_dir,
        run_id=run_id,
        module_version=args.module_version,
        manifest_hash="",   # Will be updated after MV completes.
    )

    # Install timeout (OC-01). 600 seconds.
    _install_timeout(600, fh)

    # -----------------------------------------------------------------------
    # CONTRACT CHECK (DIM-04): --module-version must equal EXPECTED_MODULE_VERSION.
    # -----------------------------------------------------------------------
    if args.module_version != EXPECTED_MODULE_VERSION:
        fh.handle(
            failure_type_id="CONTRACT_VIOLATION",
            detail=(
                f"--module-version must be '{EXPECTED_MODULE_VERSION}'. "
                f"Received: '{args.module_version}'. Hard failure per DIM-04."
            ),
        )

    # -----------------------------------------------------------------------
    # STAGE 1: MANIFEST VALIDATOR (MV) -- must complete before any other stage.
    # -----------------------------------------------------------------------
    mv = ManifestValidator(manifest_path)
    try:
        manifest_hash, thresholds = mv.validate()
    except RuntimeError as exc:
        fh.handle_from_exception(exc)

    # Update FailureHandler with real manifest hash.
    fh._manifest_hash = manifest_hash

    # -----------------------------------------------------------------------
    # STAGE 2: INPUT VECTOR GENERATOR (IVG)
    # -----------------------------------------------------------------------
    ivg     = InputVectorGenerator()
    vectors = ivg.generate()

    # -----------------------------------------------------------------------
    # STAGE 3: EXECUTION RECORDER (ER) or RecordLoader in cross-session replay.
    # -----------------------------------------------------------------------
    if args.cross_session_replay:
        if args.prior_record_path is None:
            fh.handle(
                failure_type_id="CONTRACT_VIOLATION",
                detail="--prior-record-path is required with --cross-session-replay.",
            )
        loader = RecordLoader()
        try:
            er_records = loader.load(
                filepath=Path(args.prior_record_path),
                module_version=args.module_version,
            )
        except RuntimeError as exc:
            fh.handle_from_exception(exc)
    else:
        recorder = ExecutionRecorder(
            manifest_hash=manifest_hash,
            module_version=args.module_version,
            run_id=run_id,
            stage="ER",
        )
        try:
            er_records = recorder.record_all(vectors)
        except RuntimeError as exc:
            fh.handle_from_exception(exc)

    # Serialize ER records to disk.
    serializer = RecordSerializer()
    try:
        er_path = serializer.serialize(er_records, runs_dir, run_id, "ER")
    except Exception as exc:
        fh.handle("HARNESS_INTERNAL_ERROR", f"Failed to serialize ER records: {exc}")

    # -----------------------------------------------------------------------
    # STAGE 4: REPLAY ENGINE (RE)
    # -----------------------------------------------------------------------
    re_engine = ReplayEngine(
        manifest_hash=manifest_hash,
        module_version=args.module_version,
        run_id=run_id,
    )
    try:
        re_records = re_engine.replay_all(vectors)
    except RuntimeError as exc:
        fh.handle_from_exception(exc)

    # Serialize RE records to disk.
    try:
        re_path = serializer.serialize(re_records, runs_dir, run_id, "RE")
    except Exception as exc:
        fh.handle("HARNESS_INTERNAL_ERROR", f"Failed to serialize RE records: {exc}")

    # -----------------------------------------------------------------------
    # STAGE 5: BIT-LEVEL COMPARATOR (BIC)
    # -----------------------------------------------------------------------
    bic = BitComparator()
    try:
        report = bic.compare(er_records, re_records)
    except RuntimeError as exc:
        fh.handle_from_exception(exc)

    # Check BIC result before proceeding to CCV.
    if not report.passed:
        first_mismatch = report.mismatches[0] if report.mismatches else None
        detail = (
            f"{len(report.mismatches)} field mismatch(es) detected. "
            f"First: vector={first_mismatch.vector_id if first_mismatch else 'N/A'} "
            f"field={first_mismatch.field_name if first_mismatch else 'N/A'}."
        )
        failure_type = (
            first_mismatch.failure_type
            if first_mismatch else "DETERMINISM_BREACH"
        )
        fh.handle(
            failure_type_id=failure_type,
            detail=detail,
            vector_id=first_mismatch.vector_id if first_mismatch else "",
            field_name=first_mismatch.field_name if first_mismatch else "",
        )

    # -----------------------------------------------------------------------
    # STAGE 6: CLIP-CHAIN VERIFIER (CCV)
    # -----------------------------------------------------------------------
    ccv = ClipVerifier(
        shock_exposure_cap=float(thresholds["shock_exposure_cap"]),
        max_drawdown_threshold=float(thresholds["max_drawdown_threshold"]),
        vol_compression_trigger=float(thresholds["vol_compression_trigger"]),
    )
    try:
        violations, notes = ccv.verify(er_records)
    except RuntimeError as exc:
        fh.handle_from_exception(exc)

    report = ccv.merge_into_report(report, violations, notes)

    if violations:
        fh.handle(
            failure_type_id="FREEZE_VIOLATION",
            detail=f"Clip chain violations detected: {violations[0][:200]}",
        )

    # -----------------------------------------------------------------------
    # PASS: Write pass record and exit 0.
    # -----------------------------------------------------------------------
    ts = _now_iso()
    pass_record = {
        "result":            "PASS",
        "run_id":            run_id,
        "harness_version":   HARNESS_VERSION,
        "module_version":    args.module_version,
        "manifest_hash":     manifest_hash,
        "vectors_executed":  len(er_records),
        "bic_passed":        True,
        "ccv_passed":        True,
        "ccv_notes":         list(report.notes),
        "er_record_path":    str(er_path),
        "re_record_path":    str(re_path),
        "timestamp_iso":     ts,
    }

    ts_compact = ts.replace(":", "").replace("-", "")[:16]
    pass_filename = f"{run_id}_PASS_{ts_compact}.json"
    pass_filepath = runs_dir / pass_filename
    try:
        with open(pass_filepath, "w", encoding="utf-8") as f:
            json.dump(pass_record, f, indent=4)
    except Exception as exc:
        fh.handle("HARNESS_INTERNAL_ERROR", f"Failed to write pass record: {exc}")

    # HFP-05: Stdout contains only the pass/fail summary.
    print(
        f"HARNESS RESULT: PASS\n"
        f"Run ID:          {run_id}\n"
        f"Harness version: {HARNESS_VERSION}\n"
        f"Module version:  {args.module_version}\n"
        f"Vectors:         {len(er_records)}\n"
        f"Manifest hash:   {manifest_hash[:16]}...\n"
        f"Pass record:     {pass_filepath}\n"
        f"Timestamp:       {ts}"
    )

    # Cancel timeout alarm before normal exit.
    try:
        signal.alarm(0)
    except AttributeError:
        pass

    sys.exit(0)


def run_harness(
    manifest_path:  str,
    module_version: str,
    runs_dir:       str,
) -> int:
    """
    Programmatic entry point for the Deterministic Verification Harness.

    Executes the harness exactly as if invoked via:
        python -m jarvis.verification.run_harness \
            --manifest-path <manifest_path> \
            --module-version <module_version> \
            --runs-dir <runs_dir>

    Uses subprocess to invoke the module in a child process, preserving
    all exit-code semantics (HFP-01) and the single-threaded, deterministic
    pipeline contract (EEP-01) without duplicating main() logic.

    Args:
        manifest_path:  Path string to THRESHOLD_MANIFEST.json (MVI-01).
        module_version: Risk Engine module version to verify (DIM-04).
        runs_dir:       Path string to the output run records directory (RSF-05).

    Returns:
        int: Process exit code from the harness subprocess.
            0 -- All verification passes passed. FREEZE compliant.
            1 -- FREEZE_VIOLATION or DETERMINISM_BREACH.
            2 -- INTEGRITY_FAILURE (manifest issues).
            3 -- DATA_CORRUPTION or CONTRACT_VIOLATION.
            4 -- Internal harness error.
    """
    import subprocess  # stdlib; isolated here to avoid top-level import side-effects

    cmd = [
        sys.executable,
        "-m", "jarvis.verification.run_harness",
        "--manifest-path",  manifest_path,
        "--module-version", module_version,
        "--runs-dir",       runs_dir,
    ]
    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    main()
