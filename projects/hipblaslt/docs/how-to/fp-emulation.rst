.. meta::
  :description: hipBLASLt FP32/FP64 Fixed-Point Emulation (Ozaki Scheme II)
  :keywords: hipBLASLt, ROCm, library, API, tool, FP64 emulation, FP32 emulation, fixed-point emulation

.. _fp-emulation:

*******************************
FP32/FP64 Fixed-Point Emulation
*******************************

Fixed-Point Emulation uses INT8 Tensor Core GEMMs with the Ozaki Scheme II
algorithm to emulate FP64 or FP32 matrix-multiply operations at high accuracy.
This can achieve higher throughput than native DGEMM on hardware where INT8
peak FLOPS greatly exceeds FP64 peak FLOPS.

Background
==========

The algorithm (Ozaki, Uchino, Imamura, arXiv:2504.08009) decomposes each
floating-point input matrix into a sequence of INT8 representations using
per-row and per-column shift values. A Chinese Remainder Theorem (CRT)
reconstruction step recovers the full-precision result from a set of INT8
GEMM outputs.

The Adaptive Precision Detection (ADP) mode automatically selects the minimum
number of moduli (INT8 GEMM passes) needed to achieve the target precision for
the given input data, based on the dynamic range of the inputs.

Supported types
===============

* **FP64** (``HIP_R_64F``): enabled via ``HIPBLASLT_EMULATE_DOUBLE_PRECISION=1``.
* **FP32** (``HIP_R_32F``): enabled via ``HIPBLASLT_EMULATE_SINGLE_PRECISION=1``.

Only non-batched (``batch_count=1``) GEMMs are supported.

Environment variables
=====================

Enable / disable
----------------

.. code-block:: bash

   # Enable FP64 emulation (matching cuBLAS naming)
   export HIPBLASLT_EMULATE_DOUBLE_PRECISION=1

   # Enable FP32 emulation (matching cuBLAS naming)
   export HIPBLASLT_EMULATE_SINGLE_PRECISION=1

Strategy
--------

.. code-block:: bash

   # performant (default): emulate only when predicted to be faster than native GEMM.
   # eager: always emulate regardless of problem size.
   export HIPBLASLT_EMULATION_STRATEGY=performant|eager

Precision targets (ADP)
-----------------------

.. code-block:: bash

   # ADP precision target for FP64 in mantissa bits [1..52].
   # Default 52 = full IEEE 754 FP64 precision.  Lower values → fewer moduli → faster.
   export HIPBLASLT_EMULATION_FP64_MANTISSA_BIT_COUNT=52

   # ADP precision target for FP32 in mantissa bits [1..23].
   # Default 23 = full IEEE 754 FP32 precision.  Lower values → fewer moduli → faster.
   export HIPBLASLT_EMULATION_FP32_MANTISSA_BIT_COUNT=23

   # Legacy tolerance for FP64 (kept for backward compatibility).
   # Converts a positive floating-point tolerance to a mantissa-bit count.
   # If HIPBLASLT_EMULATION_FP64_MANTISSA_BIT_COUNT is also set, that takes precedence.
   export HIPBLASLT_EMULATION_TOLERANCE=1e-16

Moduli count
------------

.. code-block:: bash

   # Force a fixed moduli count [2..20], bypassing ADP.
   # WARNING: bypasses the adaptive precision check; accuracy not guaranteed.
   # Recommended usage: do NOT set this; rely on the default ADP mode instead.
   export HIPBLASLT_EMULATION_NUM_MODULI=16

Special values
--------------

.. code-block:: bash

   # Bitmask controlling NaN/Inf detection: bit 0 = Inf, bit 1 = NaN.  Default 0x3.
   # Set to 0 to skip special-value detection (faster, but unsupported inputs silently corrupt output).
   export HIPBLASLT_EMULATION_SPECIAL_VALUES_SUPPORT_MASK=0x3

Profiling
---------

.. code-block:: bash

   # Append per-call profiling CSV rows to the given file path.
   export HIPBLASLT_EMULATION_PROFILE=/path/to/profile.csv

Setting precedence
==================

For each parameter, the precedence order (highest to lowest) is:

1. **Handle setter** (``hipblasLtSet*`` functions) — permanently overrides env vars
   for the lifetime of that handle.
2. **Environment variable** — read once at first use, process-wide default.
3. **Built-in default** — ADP mode, S_MAX=20 moduli, special-values mask 0x3.

ADP typical moduli counts
=========================

* **FP64**: s ≈ 16 is typical for full-precision emulation (52 mantissa bits).
* **FP32**: s ≈ 8 is typical for full-precision emulation (23 mantissa bits).

These are starting points; ADP selects the exact ``effective_s`` from the input
data at runtime based on the dynamic range of each matrix.

Usage
=====

FP64 emulation
--------------

.. code-block:: bash

   export HIPBLASLT_EMULATE_DOUBLE_PRECISION=1

FP32 emulation
--------------

.. code-block:: bash

   export HIPBLASLT_EMULATE_SINGLE_PRECISION=1

Both types can be enabled simultaneously. Each GEMM call dispatches on
``type_a`` to select the appropriate settings.

Workspace
=========

Emulation requires a workspace buffer. Query the required size with:

.. code-block:: c

   size_t ws = hipblasLtEmulationWorkspaceSize(handle, opA, opB, m, n, k);
   void* workspace;
   hipMalloc(&workspace, ws);

Pass the workspace via the ``FixedPointEmulationSettings`` struct to
``fp64EmulatedGemm`` or ``fp32EmulatedGemm``.

Limitations
===========

* Only ``batch_count=1`` (non-batched) GEMMs are supported.
* Performance is hardware-specific; use ``HIPBLASLT_EMULATION_STRATEGY=performant``
  (the default) to only emulate when it outperforms native GEMM.
* ADP may fall back to native GEMM for matrices with extreme dynamic range
  (e.g., condition number ≫ 2^{2·log2P_20}) by returning
  ``rocblaslt_status_invalid_value``; callers should handle this gracefully.
