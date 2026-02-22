@echo off
:: =============================================================================
:: JARVIS v6.1.0 -- CI GATE (Windows)
:: File:   run_ci.bat
:: =============================================================================
::
:: PURPOSE
:: -------
:: Windows-native CI entry point.
:: Delegates to scripts/run_ci_checks.py and propagates the exit code.
::
:: Usage:
::   run_ci.bat
::
:: Exit codes:
::   0 -- All CI stages passed (pytest + DVH gate).
::   1 -- pytest stage failed (tests or coverage < 90%).
::   2 -- DVH gate stage failed.
::
:: Compatible with: Windows CMD, batch pipelines, GitHub Actions (windows-latest).
:: =============================================================================

echo ========================================================================
echo JARVIS CI GATE (Windows)
echo ========================================================================

python scripts\run_ci_checks.py
if %errorlevel% neq 0 (
    echo.
    echo CI GATE: FAIL  -- errorlevel=%errorlevel%
    echo Merge BLOCKED.
    exit /b 1
)

echo.
echo CI GATE: PASS
exit /b 0
