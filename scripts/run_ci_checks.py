#!/usr/bin/env python3
# =============================================================================
# JARVIS v6.1.0 -- CI CHECKS RUNNER
# File:   scripts/run_ci_checks.py
# =============================================================================
#
# PURPOSE
# -------
# Runs full CI gate in two sequential stages:
#   Stage 1: pytest (all tests + coverage enforcement ≥ 90%)
#   Stage 2: DVH gate (deterministic verification harness)
#
# Exit codes:
#   0 -- All stages passed.
#   1 -- Stage 1 (pytest) failed.
#   2 -- Stage 2 (DVH gate) failed.
#
# Usage:
#   python scripts/run_ci_checks.py
#
# No external dependencies beyond stdlib and the JARVIS package.
# Deterministic: no random state, no timestamps, no side effects outside
# subprocess invocations and stdout/stderr writes.
# =============================================================================

from __future__ import annotations

import subprocess
import sys
import pathlib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).parent.parent
_PYTHON    = sys.executable


def _separator(char: str = "=", width: int = 72) -> str:
    return char * width


def _run(cmd: list[str], label: str) -> int:
    """
    Run a subprocess command, stream stdout/stderr live, return exit code.
    """
    print(_separator())
    print(f"CI STAGE: {label}")
    print(f"CMD:      {' '.join(cmd)}")
    print(_separator("-"))
    sys.stdout.flush()

    proc = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
    )
    return proc.returncode


def main() -> int:
    print(_separator())
    print("JARVIS CI GATE -- starting")
    print(_separator())
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Stage 1: pytest
    # pytest.ini provides: --cov=jarvis --cov-report=term-missing
    #                      --cov-fail-under=90
    # A non-zero exit code here means either tests failed or coverage < 90%.
    # ------------------------------------------------------------------
    pytest_rc = _run(
        [_PYTHON, "-m", "pytest"],
        "pytest (tests + coverage ≥ 90%)",
    )

    if pytest_rc != 0:
        print(_separator())
        print(f"CI RESULT: FAIL  [stage=pytest  exit_code={pytest_rc}]")
        print("Merge BLOCKED: pytest stage did not pass.")
        print(_separator())
        sys.stdout.flush()
        return 1

    print(_separator("-"))
    print("CI STAGE pytest: PASS")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Stage 2: DVH gate
    # Runs jarvis/verification/ci_dvh_gate.py via its module entry point.
    # A non-zero exit code means the Deterministic Verification Harness
    # detected a freeze violation, determinism breach, or integrity failure.
    # ------------------------------------------------------------------
    dvh_rc = _run(
        [_PYTHON, "-m", "jarvis.verification.ci_dvh_gate"],
        "DVH gate (Deterministic Verification Harness)",
    )

    if dvh_rc != 0:
        print(_separator())
        print(f"CI RESULT: FAIL  [stage=dvh  exit_code={dvh_rc}]")
        print("Merge BLOCKED: DVH gate did not pass.")
        print(_separator())
        sys.stdout.flush()
        return 2

    print(_separator("-"))
    print("CI STAGE dvh: PASS")

    # ------------------------------------------------------------------
    # All stages passed.
    # ------------------------------------------------------------------
    print(_separator())
    print("CI RESULT: PASS  [stages=pytest,dvh]")
    print("Merge permitted.")
    print(_separator())
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
