.. meta::
   :description: How to use FP64 GEMM emulation via Ozaki Scheme II with the hipBLASLt library
   :keywords: hipBLASLt, ROCm, library, API, FP64, emulation, Ozaki, INT8, tensor cores

.. _fp64-emulation:

*************************************
Using FP64 emulation with hipBLASLt
*************************************

hipBLASLt can emulate FP64 GEMM operations on hardware that provides high-throughput INT8 tensor cores
but limited native FP64 throughput.
The emulation is based on *Ozaki Scheme II (accurate mode)*, which decomposes the FP64 input matrices
into a series of scaled INT8 representations, performs multiple INT8 GEMMs on the tensor cores, and
reconstructs the result using Chinese Remainder Theorem (CRT) arithmetic with double-double accumulation.

For algorithmic details and accuracy analysis, see:

*  K. Ozaki, S. Uchino, and T. Imamura, "GEMMul8: High-performance FP64 GEMM via INT8 Tensor Cores", arXiv:2504.08009, 2025.
   Available at `<https://arxiv.org/abs/2504.08009>`_.
*  GEMMul8 open-source implementation: `<https://github.com/RIKEN-RCCS/GEMMul8>`_.

Overview
=========

With the default ADP configuration, the emulated result is equivalent in accuracy to native FP64
GEMM for all well-conditioned inputs.
The algorithm adapts automatically to the problem size: a preliminary INT8 GEMM is used to tighten
the per-row and per-column scaling shifts before the main computation, ensuring that the CRT
capacity is used efficiently regardless of the dynamic range of the input data.

Emulation is applied only to FP64 matrix-matrix multiplications (``HIP_R_64F`` data type with
``HIPBLAS_COMPUTE_64F``).
All other GEMM configurations pass through to the normal hipBLASLt code path.

Enabling FP64 emulation
========================

FP64 emulation can be enabled either globally via an environment variable or per-handle via the API.

**Using the environment variable (process-wide)**

Set ``HIPBLASLT_EMULATE_DOUBLE_PRECISION=1`` before launching your application:

.. code-block:: bash

   export HIPBLASLT_EMULATE_DOUBLE_PRECISION=1
   ./my_application

**Using the API (per-handle)**

.. code-block:: c

   hipblasLtHandle_t handle;
   hipblasLtCreate(&handle);

   // Enable emulation for this handle.
   hipblasLtSetEmulationEnabled(handle, true);

   // Optionally set the strategy to EAGER to emulate all FP64 GEMMs.
   hipblasLtSetEmulationStrategy(handle, HIPBLASLT_EMULATION_STRATEGY_EAGER);

Once emulation is enabled, the normal ``hipblasLtMatmul`` call is used — no other code changes
are required.

Configuring the emulation strategy
====================================

The emulation strategy controls when hipBLASLt applies emulation.

.. csv-table::
   :header: "Strategy", "Environment variable value", "API constant", "Description"
   :widths: 15, 25, 40, 60

   "Default", "*(inherits HIPBLASLT_EMULATION_STRATEGY)*", "``HIPBLASLT_EMULATION_STRATEGY_DEFAULT``", "Defer to the process-wide environment variable."
   "Performant", "``performant``", "``HIPBLASLT_EMULATION_STRATEGY_PERFORMANT``", "Emulate only when the arithmetic-intensity heuristic predicts that INT8 tensor core throughput exceeds native FP64 throughput. Suitable for large GEMMs."
   "Eager", "``eager``", "``HIPBLASLT_EMULATION_STRATEGY_EAGER``", "Emulate all supported FP64 GEMMs, regardless of problem size. Use this when consistent accuracy is required across all sizes."

Set the strategy via environment variable:

.. code-block:: bash

   export HIPBLASLT_EMULATION_STRATEGY=eager

Or via the API:

.. code-block:: c

   hipblasLtSetEmulationStrategy(handle, HIPBLASLT_EMULATION_STRATEGY_EAGER);

Configuring the number of CRT moduli
=========================================

The emulation uses a set of coprime moduli to represent the matrix entries in the CRT domain.
More moduli mean higher CRT capacity (more mantissa bits) at the cost of more INT8 GEMMs.
The default is ADP mode, which adaptively selects the minimum number of moduli needed to achieve
FP64 accuracy on each call. In ADP mode up to 16 moduli (~125 mantissa bits) may be used.

**ADP mode (default — library selects number of moduli automatically)**

ADP mode is the default and the recommended setting. The library chooses the fewest moduli that
maintain accuracy for the given input data. No configuration is required.

**Fixed mode via environment variable**

Set ``HIPBLASLT_EMULATION_NUM_MODULI`` to a fixed moduli count in [2..18]:

.. code-block:: bash

   # 7 moduli (~55.7 CRT bits)
   export HIPBLASLT_EMULATION_NUM_MODULI=7

   # 16 moduli (~125.4 CRT bits) — exceeds IEEE 754 double precision (53 bits)
   export HIPBLASLT_EMULATION_NUM_MODULI=16

.. warning::

   Fixed mode does **not** guarantee accuracy for all inputs. CRT sign-flip errors can occur when
   the input's dynamic range exceeds the capacity of the selected moduli set. Use ADP mode unless
   you have validated that a fixed count is safe for your specific inputs.

**Using the API**

.. code-block:: c

   // ADP mode (default): library adaptively selects the number of moduli.
   hipblasLtSetEmulationNumModuli(handle, -1);

   // Fixed mode: use exactly N moduli (N in [2..18]).
   // Warning: does not guarantee accuracy for all inputs.
   hipblasLtSetEmulationNumModuli(handle, 16);

Configuring Inf/NaN detection
================================

By default, the emulation checks for Inf and NaN values in the input matrices.
On each call, a small device-to-host synchronization is performed to read the detection flag.
This can add latency for very small GEMMs.
If your application guarantees clean (finite, non-NaN) inputs, you can disable the check:

.. code-block:: bash

   # Disable both Inf and NaN detection (bit 0 = Inf, bit 1 = NaN).
   export HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=0

Or via the API:

.. code-block:: c

   hipblasLtSetEmulationSpecialValuesSupport(handle, 0u);

Numerical behavior and limitations
=====================================

Inf and NaN inputs
------------------

By default (``HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=3``), the emulation checks every
element of the input matrices for Inf and NaN before computing.
If any Inf or NaN is found, the emulation falls back internally to native FP64 DGEMM, which
propagates the Inf or NaN to the output in the standard IEEE 754 manner.
``hipblasLtMatmul`` returns ``HIPBLAS_STATUS_SUCCESS``; non-finite values appear in D exactly
as they would without emulation.
The per-call device-to-host synchronization for the check can be skipped by setting mask ``0``
if your application guarantees finite inputs.

Subnormal inputs (flush-to-zero semantics)
-------------------------------------------

The emulation applies **flush-to-zero (FTZ) semantics to subnormal inputs**.
Subnormal FP64 values (magnitudes between roughly ``5×10⁻³²⁴`` and ``2.2×10⁻³⁰⁸``) are not
detected by the special-values mask and are **silently treated as zero** during the INT8
extraction step.

Specifically:

*  During the preliminary per-row shift computation, any row whose maximum absolute value is
   below ``10⁻³⁰⁰`` is treated as a unit-magnitude row (the subnormal or zero guard prevents
   ``log2(0)``).
*  During the final extraction, each element is scaled by ``2^sft`` and then truncated to the
   nearest integer.
   For subnormal values, this product is still subnormal (less than ``2⁻¹⁰¹⁶``), so the integer
   truncation returns **zero**.

**Consequence**: if the true result of a dot product is a subnormal FP64 value, the emulated
result may be returned as **0.0** instead of the correct subnormal.
The absolute error is at most ``2.2×10⁻³⁰⁸`` (the minimum normal double).
Native FP64 DGEMM handles gradual underflow correctly and will return the true subnormal value.

If subnormal correctness is required, disable emulation for the affected GEMMs
(``hipblasLtSetEmulationEnabled(handle, false)``) or use the eager strategy only for the
computationally intensive part of your workload where subnormal outputs are unlikely.

Extreme dynamic range — ADP overflow fallback
----------------------------------------------

Native FP64 DGEMM can produce ``Inf`` from finite inputs when the magnitude of the inner
products exceeds the FP64 range (approximately when
``max(|A|) × max(|B|) × k > 1.8 × 10³⁰⁸``).
The emulation works through scaled INT8 integer arithmetic and cannot itself produce ``Inf``,
so it must detect these cases and fall back rather than returning a wrong finite result.

In **ADP mode** (the default, set with ``hipblasLtSetEmulationNumModuli(handle, -1)``), the library
estimates the required CRT capacity from the preliminary INT8 GEMM result.
For inputs that would overflow native FP64 DGEMM, the required CRT capacity is far above the
maximum supported (~140 bits, 18 moduli), so ADP detects this and falls back internally to
native FP64 DGEMM.
The native path then correctly computes the result, including ``Inf`` when appropriate.
``hipblasLtMatmul`` returns ``HIPBLAS_STATUS_SUCCESS``.
A rate-limited warning is also printed to ``stderr`` (at most 5 times per process).

In **fixed-s mode** (set with ``hipblasLtSetEmulationNumModuli(handle, N)`` for N in [2..18]),
no CRT overflow check is performed.
For inputs that would overflow native FP64 DGEMM, fixed-s emulation silently produces a wrong
finite result instead of ``Inf``.
Users who opt into fixed-s mode should ensure that their inputs do not cause DGEMM overflow
(for example by keeping input magnitudes well below ``2^500``).

Environment variables reference
==================================

The following environment variables apply process-wide to all hipBLASLt handles.
When both are set, FP64 emulation environment variables take precedence over
per-handle API settings to match the cuBLAS environment-variable contract.

.. csv-table::
   :header: "Environment Variable", "Default", "Description"
   :widths: 50, 15, 80

   "``HIPBLASLT_EMULATE_DOUBLE_PRECISION``", "``0``", "Set to ``1`` to enable FP64 emulation for all handles in the process."
   "``HIPBLASLT_EMULATION_STRATEGY``", "``performant``", "Controls when emulation is applied: ``performant`` (arithmetic-intensity heuristic) or ``eager`` (always)."
   "``HIPBLASLT_EMULATION_NUM_MODULI``", "*(unset → ADP)*", "Fixed number of CRT moduli to use [2..18]. When unset, ADP mode is active and the library selects the moduli count adaptively per call. Warning: fixed mode does not guarantee accuracy for all inputs."
   "``HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK``", "``3``", "Bitmask controlling Inf/NaN detection. Bit 0 = Inf detection; bit 1 = NaN detection. Set to ``0`` to disable both and avoid the associated device-to-host synchronization."

API reference
===============

The following functions control FP64 emulation on a per-handle basis.
Include ``<hipblaslt/hipblaslt.h>`` to use these APIs.

hipblasLtSetEmulationEnabled
-----------------------------

.. code-block:: c

   hipblasStatus_t hipblasLtSetEmulationEnabled(hipblasLtHandle_t handle, bool enabled);

Enables (``true``) or disables (``false``) FP64 emulation for the specified handle.
When ``false``, emulation is suppressed for all GEMMs issued through this handle, even if the
process-wide environment variable is set.

hipblasLtSetEmulationStrategy / hipblasLtGetEmulationStrategy
--------------------------------------------------------------

.. code-block:: c

   hipblasStatus_t hipblasLtSetEmulationStrategy(hipblasLtHandle_t            handle,
                                                 hipblasLtEmulationStrategy_t strategy);

   hipblasStatus_t hipblasLtGetEmulationStrategy(hipblasLtHandle_t             handle,
                                                 hipblasLtEmulationStrategy_t* strategy);

Sets or queries the emulation strategy for a handle.
Valid values for ``hipblasLtEmulationStrategy_t``:

*  ``HIPBLASLT_EMULATION_STRATEGY_DEFAULT`` — inherit from ``HIPBLASLT_EMULATION_STRATEGY``.
*  ``HIPBLASLT_EMULATION_STRATEGY_PERFORMANT`` — apply emulation only when the arithmetic-intensity
   heuristic predicts a throughput benefit.
*  ``HIPBLASLT_EMULATION_STRATEGY_EAGER`` — apply emulation to all supported FP64 GEMMs.

hipblasLtSetEmulationNumModuli
------------------------------

.. code-block:: c

   hipblasStatus_t hipblasLtSetEmulationNumModuli(hipblasLtHandle_t handle, int numModuli);

Sets the number of CRT moduli used per emulation call.

*  ``numModuli = -1`` — ADP mode (default). The library adaptively selects the minimum number of
   moduli needed to achieve FP64 accuracy on each call based on the input data.
*  ``numModuli`` in [2..18] — FIXED mode. Exactly that many moduli are used regardless of input.

.. warning::

   FIXED mode does **not** guarantee numerical accuracy. CRT sign-flip errors can occur for inputs
   with large dynamic range. A one-time per-process warning is printed when FIXED mode is selected.
   Use ADP mode (the default) for reliable results.

Notable CRT capacity values:

*  ``7`` → ~55.7 CRT bits
*  ``10`` → ~79.2 CRT bits
*  ``16`` → ~125.4 CRT bits (exceeds IEEE 754 double precision)
*  ``18`` → ~140.4 CRT bits (maximum)

hipblasLtSetEmulationSpecialValuesSupport
------------------------------------------

.. code-block:: c

   hipblasStatus_t hipblasLtSetEmulationSpecialValuesSupport(hipblasLtHandle_t handle,
                                                              uint32_t          mask);

Sets the bitmask that controls Inf/NaN detection for a handle.

*  Bit 0 — detect Inf values.
*  Bit 1 — detect NaN values.
*  Default mask: ``3`` (both enabled).
*  Set to ``0`` to disable detection and avoid the associated device-to-host synchronization.

hipblasLtEmulationWorkspaceSize
---------------------------------

.. code-block:: c

   size_t hipblasLtEmulationWorkspaceSize(hipblasLtHandle_t  handle,
                                          hipblasOperation_t opA,
                                          hipblasOperation_t opB,
                                          int64_t            m,
                                          int64_t            n,
                                          int64_t            k);

Returns the GPU workspace size in bytes required by the emulation for a problem of size
*m* × *n* × *k*, given the transpose modes ``opA`` and ``opB``.
The ``handle`` provides the device selection and the configured moduli count (as set via
``hipblasLtSetEmulationNumModuli``).
Returns ``0`` if emulation is not supported for the current device or if the handle is
configured for native-only mode.

This workspace must be provided to ``hipblasLtMatmul`` via the heuristic preference object
(``HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES``).
Alternatively, call ``hipblasLtMatmulAlgoGetHeuristic`` with a sufficiently large workspace budget;
the heuristic result will report the exact bytes required in ``hipblasLtMatmulHeuristicResult_t::workspaceSize``.
