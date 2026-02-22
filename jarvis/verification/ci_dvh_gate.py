#!/usr/bin/env python3
# =============================================================================
# JARVIS v6.1.0 -- DVH CI GATE
# File:   jarvis/verification/ci_dvh_gate.py
# Version: 1.0.0
# =============================================================================
#
# PURPOSE
# -------
# CI enforcement script. Runs the Deterministic Verification Harness and
# exits with code 0 (PASS) or 1 (FAIL / ERROR).
#
# Intended for CI integration:
#   python -m jarvis.verification.ci_dvh_gate
#
# Exit codes:
#   0 -- DVH PASS: all vectors pass, BIC and CCV pass.
#   1 -- DVH FAIL or ERROR: CI must block merge.
#
# No I/O beyond stdout/stderr and the runs/ directory (via run_harness).
# No network calls. No external dependencies beyond stdlib and numpy.
# =============================================================================

from __future__ import annotations

import json
import pathlib
import sys
import importlib


_MANIFEST_PATH   = pathlib.Path(__file__).parent.parent / "risk" / "THRESHOLD_MANIFEST.json"
_MODULE_VERSION  = "6.1.0"
_RUNS_DIR        = pathlib.Path(__file__).parent / "runs"


def main() -> int:
    """
    Run DVH and return exit code.

    Returns:
        0 if DVH result is PASS.
        1 if DVH result is FAIL, ERROR, or an exception occurs.
    """
    try:
        # Import run_harness dynamically to avoid circular imports at module level.
        run_harness_mod = importlib.import_module("jarvis.verification.run_harness")
        run_fn = getattr(run_harness_mod, "run_harness", None)
        if run_fn is None:
            print("CI-DVH-GATE ERROR: run_harness.run_harness() not found.", file=sys.stderr)
            return 1

        result = run_fn(
            manifest_path=str(_MANIFEST_PATH),
            module_version=_MODULE_VERSION,
            runs_dir=str(_RUNS_DIR),
        )

        if result == 0:
            print(f"CI-DVH-GATE: DVH result=PASS. Merge permitted.")
            return 0
        else:
            print(f"CI-DVH-GATE: DVH result={result}. Merge BLOCKED.", file=sys.stderr)
            return 1

    except Exception as exc:  # noqa: BLE001
        print(f"CI-DVH-GATE EXCEPTION: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
