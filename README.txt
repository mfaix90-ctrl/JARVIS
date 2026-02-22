================================================================================
MASP -- MULTI-ASSET STRATEGY PLATFORM
JARVIS Decision Quality Platform
Version: 6.1.0 / Harness: 1.0.0
Status: FREEZE
================================================================================

SYSTEM CLASSIFICATION (P0 -- IMMUTABLE)
  This is a pure analysis and strategy research platform.
  It is NOT a trading system. It is NOT an execution engine.
  It does NOT trigger live orders. It does NOT manage real capital.
  It does NOT integrate with any broker API.
  All computations operate on simulated positions only.

DIRECTORY LAYOUT
  MASP/
    jarvis/
      __init__.py
      core/
        __init__.py
        regime.py               -- Canonical regime enums and HierarchicalRegime
      risk/
        __init__.py
        risk_engine.py          -- RiskEngine FAS v6.1.0
        THRESHOLD_MANIFEST.json -- Hash-protected constants manifest
      utils/
        __init__.py
        constants.py            -- JOINT_RISK_MULTIPLIER_TABLE and platform constants
      verification/             -- Deterministic Verification Harness v1.0.0
        __init__.py
        harness_version.py
        run_harness.py          -- Entry point
        manifest_validator.py
        input_vector_generator.py
        execution_recorder.py
        replay_engine.py
        bit_comparator.py
        clip_verifier.py
        failure_handler.py
        data_models/
          __init__.py
          input_vector.py
          execution_record.py
          comparison_report.py
          failure_record.py
        vectors/
          __init__.py
          vector_definitions.py
        storage/
          __init__.py
          record_serializer.py
          record_loader.py
        config/
          harness_config.json
        runs/                   -- Runtime-generated records (write-once)
    README.txt

REQUIREMENTS
  Python >= 3.10
  numpy >= 1.24.0
  scipy >= 1.10.0  (for percentile in compute_expected_drawdown fallback)

  Standard library only for the Deterministic Verification Harness.
  The harness has zero third-party runtime dependencies.

QUICK IMPORT CHECK
  python -c "from jarvis.risk.risk_engine import RiskEngine; print('OK')"

STANDARD INVOCATION (Risk Engine)
  from jarvis.risk.risk_engine import RiskEngine
  from jarvis.core.regime import GlobalRegimeState
  engine = RiskEngine()
  result = engine.assess(returns_history=[...], current_regime=GlobalRegimeState.RISK_ON, meta_uncertainty=0.2)

HARNESS INVOCATION
  python -m jarvis.verification.run_harness --manifest-path jarvis/risk/THRESHOLD_MANIFEST.json --module-version 6.1.0 --runs-dir jarvis/verification/runs

GOVERNANCE
  All numeric constants are hash-protected in THRESHOLD_MANIFEST.json.
  No constant may be changed without a version bump and manifest update.
  Any behavioral change requires an updated FAS document.
  The Deterministic Verification Harness enforces FREEZE compliance of FAS v6.1.0.

Authority: Risk Engine FAS v6.1.0 | Master FAS v6.1.0-G | DVH Blueprint v1.0.0
================================================================================
