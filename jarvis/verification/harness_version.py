# jarvis/verification/harness_version.py
# Harness version constant. Single authoritative definition.
# Referenced by run_harness.py, record_serializer.py, record_loader.py,
# and failure_handler.py for version stamping.
# A change to this constant requires a full harness version increment.

HARNESS_VERSION: str = "1.0.0"

# The module version this harness verifies (DIM-04).
# The --module-version CLI argument must equal this value.
EXPECTED_MODULE_VERSION: str = "6.1.0"

# Storage format version for record serialization.
STORAGE_FORMAT_VERSION: str = "1.0.0"
