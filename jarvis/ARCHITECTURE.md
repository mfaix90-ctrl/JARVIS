JARVIS ARCHITECTURE SPECIFICATION
FAS v6.1.0
===========================================================================

SECTION 1 -- LAYER STRUCTURE
===========================================================================

The JARVIS system is organized into two categories of layers: Stable Core
Layers and External Orchestration Layers.

STABLE CORE LAYERS
These layers contain canonical logic that must not be reimplemented,
extended, or modified without a signed release and version bump:

  jarvis/core/
    Single authoritative source for all regime enum definitions and the
    canonical regime transmission object. No other module may define
    regime types.

  jarvis/risk/
    Single authoritative source for the RiskEngine and all risk
    assessment arithmetic. Contains THRESHOLD_MANIFEST.json which
    hash-protects all numeric constants.

  jarvis/utils/
    Single authoritative source for all hash-protected numeric constants.
    Constants are declared once and imported everywhere. No constant may
    be redeclared or shadowed in any other module.

  jarvis/portfolio/
    Single authoritative source for portfolio allocation logic.
    allocate_positions() is the canonical allocation function.
    No other module may reimplement allocation arithmetic.

  jarvis/execution/
    Single authoritative source for the boundary adapter between risk
    exposure output and portfolio allocation.
    route_exposure_to_positions() is the canonical routing function.
    No other module may reimplement routing logic.

EXTERNAL ORCHESTRATION LAYERS
These layers may call into stable core layers but must never reimplement
any logic owned by those layers:

  jarvis/orchestrator/
    Connects RiskEngine.assess() -> exposure_weight ->
    route_exposure_to_positions() -> final positions.
    Pure delegation. No arithmetic. No regime logic.

  jarvis/backtest/
    Rolls a sliding window over time series data and delegates each
    timestep to run_full_pipeline(). Manages equity curve accumulation
    only. No risk logic. No allocation logic.

  jarvis/walkforward/
    Reserved for walk-forward validation over backtest segments.
    Must delegate entirely to jarvis/backtest/ and jarvis/orchestrator/.
    No risk logic. No allocation logic.


SECTION 2 -- IMPORT RULES
===========================================================================

The following directed import rules are enforced. An arrow A -> B means
module A is permitted to import from module B.

  jarvis/orchestrator/   -> jarvis/core/
  jarvis/orchestrator/   -> jarvis/risk/
  jarvis/orchestrator/   -> jarvis/execution/

  jarvis/backtest/       -> jarvis/core/
  jarvis/backtest/       -> jarvis/orchestrator/

  jarvis/walkforward/    -> jarvis/core/
  jarvis/walkforward/    -> jarvis/backtest/
  jarvis/walkforward/    -> jarvis/orchestrator/

  jarvis/execution/      -> jarvis/core/
  jarvis/execution/      -> jarvis/portfolio/

  jarvis/portfolio/      -> (standard library only)

  jarvis/risk/           -> jarvis/core/
  jarvis/risk/           -> jarvis/utils/

  jarvis/utils/          -> jarvis/core/

  jarvis/core/           -> (standard library only)

PROHIBITED IMPORT DIRECTIONS

  Stable core layers must never import from external orchestration layers.
  jarvis/core/       must never import from orchestrator/, backtest/, walkforward/
  jarvis/risk/       must never import from orchestrator/, backtest/, walkforward/
  jarvis/utils/      must never import from orchestrator/, backtest/, walkforward/
  jarvis/portfolio/  must never import from orchestrator/, backtest/, walkforward/
  jarvis/execution/  must never import from orchestrator/, backtest/, walkforward/

  Circular imports are prohibited in all cases without exception.

  No module outside jarvis/core/ may define, shadow, or alias any enum
  or type that is defined in jarvis/core/regime.py.

  No module outside jarvis/utils/ may declare a numeric constant that
  duplicates or shadows a constant registered in THRESHOLD_MANIFEST.json.


SECTION 3 -- DETERMINISM GUARANTEES
===========================================================================

All modules in the JARVIS system must satisfy the following determinism
guarantees without exception:

  DET-01  No stochastic operations anywhere in the system.
          No calls to random(), numpy.random, secrets, os.urandom, or
          any other source of non-deterministic values.

  DET-02  No external state reads inside any computational function.
          All inputs must be passed explicitly as function parameters.
          No function may read from module-level mutable variables,
          class-level mutable variables, or any shared mutable container
          during execution.

  DET-03  No side effects. Computational functions must not write to any
          external state, mutate any input argument, modify any module-level
          variable, or produce any observable effect beyond their return value.

  DET-04  All arithmetic operations are deterministic floating-point.
          No symbolic math, no lazy evaluation, no deferred computation.

  DET-05  All conditional branches are deterministic functions of their
          explicit inputs. No branch may depend on time, process ID,
          thread state, or any other implicit environmental value.

  DET-06  Fixed literals used as algorithm parameters must not be
          parameterised, overridden, or made configurable at runtime.
          They are part of the algorithm specification.

  DET-07  Backward compatibility is deterministic. For any call site that
          passes the same inputs as a prior version, the output must be
          bit-identical to that prior version unless a version bump with
          migration documentation has been issued.


SECTION 4 -- PROHIBITED ACTIONS
===========================================================================

The following actions are unconditionally prohibited in all modules at
all layers. There are no exceptions.

  PROHIBITED-01  Stochastic operations.
    No random number generation. No sampling. No probabilistic branching.
    No Monte Carlo. No stochastic simulation of any kind.

  PROHIBITED-02  File input and output in computational layers.
    No open(). No pathlib.Path.read*(). No pathlib.Path.write*().
    No csv, json, pickle, shelve, sqlite3, or any other I/O library call
    inside any function that performs computation or returns a result.
    File I/O is permitted only in dedicated loader modules that are
    explicitly excluded from the computational layer definition and that
    produce only immutable inputs to computational functions.

  PROHIBITED-03  Logging in computational layers.
    No calls to logging.getLogger(). No print() statements. No sys.stderr
    writes. No structured logging frameworks inside computational functions.

  PROHIBITED-04  Environment variable access.
    No os.environ. No os.getenv(). No dotenv. No configuration loaded from
    environment at runtime inside any computational function.

  PROHIBITED-05  Global mutable state.
    No module-level mutable containers that are written to during execution.
    No singleton objects that accumulate state across calls. No class-level
    mutable attributes shared across instances.

  PROHIBITED-06  Reimplementation of canonical logic.
    No module outside the canonical owner may reimplement, duplicate, or
    inline the logic of any canonical function. All callers must import
    and delegate to the canonical function.

  PROHIBITED-07  Modification of hash-protected constants.
    No numeric constant registered in THRESHOLD_MANIFEST.json may be
    changed at runtime, overridden by subclassing, or shadowed by a local
    variable of the same name within the scope of any function that uses
    the constant in arithmetic.

  PROHIBITED-08  New regime taxonomy definitions.
    No module outside jarvis/core/regime.py may define a new enum class
    that represents a regime state, correlation state, asset class, or
    any other taxonomy that duplicates or extends the canonical enums.

  PROHIBITED-09  String-based regime branching.
    No module may branch on the string value of a regime by comparing
    against a hardcoded string literal. All regime comparisons must use
    canonical enum instances imported from jarvis/core/regime.py.

  PROHIBITED-10  Architectural drift.
    No new dependency edge may be introduced between modules without
    explicit review against the import rules in Section 2. No new layer
    may be created without documenting it in this file and confirming
    compliance with all rules in Sections 2 through 7.


SECTION 5 -- DELEGATION RULE
===========================================================================

Any module that needs the result of a computation owned by another module
must import and call the canonical function that produces that result.

A module must never reproduce, inline, or approximate the logic of a
function it does not own, even if the reproduction would be functionally
equivalent.

Delegation is the only permitted form of code reuse across module
boundaries.

This rule applies at all layers. It applies to single-line computations
as well as multi-step algorithms. Simplicity of the reimplementation does
not exempt it from this rule.

If a canonical function does not exist for a required computation, the
correct action is to create a new canonical function in the appropriate
stable core layer and import it from there. Creating a local approximation
in an orchestration layer is prohibited.


SECTION 6 -- NO REIMPLEMENTATION RULE
===========================================================================

The no-reimplementation rule is a specific application of the delegation
rule. It states:

  The logic of any function defined in a stable core layer must exist in
  exactly one place in the codebase: the canonical definition in that
  stable core layer.

  Consequence for orchestration layers: orchestration layers may not
  contain any arithmetic, conditional branching, or data transformation
  that duplicates logic already present in a stable core layer, even if
  the orchestration layer applies that logic to different data shapes or
  at a different granularity.

  Consequence for testing: test files that need to replicate a computation
  to assert a result must import the canonical function and call it.
  Tests must not contain hardcoded expected values derived by manually
  executing canonical arithmetic outside the system.

  Consequence for documentation: documentation examples must call
  canonical functions. Documentation must not contain inline pseudocode
  that reimplements canonical logic.


SECTION 7 -- EXTERNAL LAYER DEFINITION
===========================================================================

An external layer is any module or package that satisfies all of the
following conditions:

  1. It resides outside jarvis/core/, jarvis/risk/, jarvis/utils/,
     jarvis/portfolio/, and jarvis/execution/.

  2. It does not define any canonical enum, canonical constant, or
     canonical computational function.

  3. It delegates all risk assessment, regime classification, allocation
     arithmetic, and exposure routing to stable core layers via import.

  4. It satisfies all determinism guarantees in Section 3.

  5. It complies with all prohibitions in Section 4.

Current external layers: jarvis/orchestrator/, jarvis/backtest/,
jarvis/walkforward/ (reserved).

A module is not automatically external simply because it resides outside
the stable core directories. A module that reimplements canonical logic
violates the no-reimplementation rule regardless of its directory location.


SECTION 8 -- STABILITY CONTRACT FOR STABLE CORE LAYERS
===========================================================================

The stable core layers -- jarvis/core/, jarvis/risk/, jarvis/utils/,
jarvis/portfolio/, jarvis/execution/ -- are governed by the following
stability contract:

  CONTRACT-01  No file inside a stable core layer may be modified without
    a signed release that increments the version number and updates
    THRESHOLD_MANIFEST.json if any hash-protected value is affected.

  CONTRACT-02  No numeric constant inside a stable core layer may be
    changed between versions without a migration document that records
    the old value, the new value, the rationale, and the impact on all
    downstream call sites.

  CONTRACT-03  No function signature inside a stable core layer may be
    changed in a backward-incompatible way without a major version bump.

  CONTRACT-04  No enum value inside jarvis/core/regime.py may be removed
    or renamed without auditing all call sites across the entire codebase
    and issuing a migration guide.

  CONTRACT-05  No new enum value may be added to jarvis/core/regime.py
    without confirming that all lookup tables and branching logic that
    reference that enum are updated to handle the new value explicitly.
    Silent fallthrough to a default case is not permitted.

  CONTRACT-06  The canonical exposure equation defined in risk_engine.py
    must remain identical across versions unless a full FAS revision is
    issued. Arithmetic modifications require a new FAS version number and
    a complete re-hash of all affected entries in THRESHOLD_MANIFEST.json.

  CONTRACT-07  The clip chain order defined in risk_engine.py -- Clip A,
    Clip B, Clip C, CRISIS dampening -- must remain identical across
    versions. Reordering constitutes a semantic change and requires a
    full FAS revision.


SECTION 9 -- ORCHESTRATOR ROLE
===========================================================================

jarvis/orchestrator/ is the canonical integration point between the risk
assessment layer and the portfolio allocation layer.

Its sole responsibility is to connect the output of RiskEngine.assess()
to the input of route_exposure_to_positions() in the correct order with
the correct field extraction.

The orchestrator must:

  1. Instantiate RiskEngine fresh per call. No instance may be reused
     across pipeline invocations. This satisfies DET-02.

  2. Call RiskEngine.assess() with all required parameters passed
     explicitly. No parameter may be sourced from module-level state.

  3. Extract exposure_weight from the returned RiskOutput. No other
     field of RiskOutput may be used by the orchestrator for computation.
     Other fields may be passed through to callers as part of a richer
     output type if a future version requires it, but such extension
     requires a version bump and must not change existing behavior.

  4. Pass exposure_weight as exposure_fraction to
     route_exposure_to_positions(). The field name mapping is fixed.
     No scaling, transformation, or clamping of exposure_weight is
     permitted between extraction and delegation.

  5. Return the positions dict unchanged from route_exposure_to_positions().
     No post-processing of positions is permitted in the orchestrator.

The orchestrator must not:

  - Contain any risk arithmetic.
  - Contain any allocation arithmetic.
  - Contain any regime branching logic.
  - Validate inputs that are already validated by downstream modules.
  - Cache any result across calls.
  - Read from or write to any external state.


SECTION 10 -- BACKTEST ROLE
===========================================================================

jarvis/backtest/ is the canonical layer for rolling-window historical
simulation over a time series of returns and prices.

Its sole responsibility is to manage the iteration over timesteps, the
slicing of the lookback window, the delegation of each timestep to
run_full_pipeline(), and the accumulation of the equity curve.

The backtest layer must:

  1. Validate that window >= 20 before beginning iteration. This mirrors
     the minimum length requirement of RiskEngine.compute_expected_drawdown().

  2. Validate that initial_capital > 0 before beginning iteration.

  3. Validate that asset_price_series and returns_series have equal length
     before beginning iteration.

  4. At each timestep t >= window, slice the lookback window as
     returns_series[t - window : t]. The slice must be exactly window
     elements. No padding, no interpolation.

  5. Delegate each timestep entirely to run_full_pipeline() by passing
     the window slice as returns_history, the current price as the sole
     entry in asset_prices, and all other parameters unchanged.

  6. Extract position_size as the scalar value from the single-entry
     positions dict returned by run_full_pipeline().

  7. Update equity as: equity * (1.0 + returns_series[t] * position_size).
     This is the only arithmetic the backtest layer may perform.
     No other formula may be used for equity updating.

  8. Append the updated equity value to the equity curve after each step.

  9. Return the equity curve as a list of floats after all timesteps are
     processed.

The backtest layer must not:

  - Contain any risk arithmetic beyond the equity update formula above.
  - Contain any allocation arithmetic.
  - Contain any regime detection or regime branching logic.
  - Reimplement any portion of run_full_pipeline().
  - Cache pipeline results across timesteps.
  - Read from or write to any file or external state.
  - Produce any output other than the equity curve list.

===========================================================================
END OF ARCHITECTURE SPECIFICATION
FAS v6.1.0
===========================================================================
