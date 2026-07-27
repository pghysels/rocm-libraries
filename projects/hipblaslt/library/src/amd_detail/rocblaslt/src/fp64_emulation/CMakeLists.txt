# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# FP64 emulation via Ozaki Scheme II (INT8 tensor cores + CRT).
#
# This translation unit is built as a standalone OBJECT library so that its
# object files can be linked PRIVATE into both the hipblaslt shared library
# (where the symbols stay hidden, leaving the public ABI unchanged) and the
# hipblaslt-test harness (which needs to call the internal entry points
# directly).  This avoids exporting internal symbols from the release .so.
add_library(hipblaslt-fp64-emulation OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/fp64_emulation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fp64_emulation_fused.cpp"
)
set_target_properties(hipblaslt-fp64-emulation PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_features(hipblaslt-fp64-emulation PRIVATE cxx_std_17)
target_link_libraries(hipblaslt-fp64-emulation PRIVATE
    hipblaslt::headers
    hip::device
    roc::${hipblas_target}
)
# TODO: These duplicate the PUBLIC definitions on the main `hipblaslt` target
# (projects/hipblaslt/CMakeLists.txt).  Fix: move them to `hipblaslt-headers`
# INTERFACE so all consumers inherit them automatically, then remove this block.
target_compile_definitions(hipblaslt-fp64-emulation PRIVATE
    ROCM_USE_FLOAT16
    __HIP_HCC_COMPAT_MODE__=1
)
target_include_directories(hipblaslt-fp64-emulation PRIVATE
    "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include"
    "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(hipblaslt PRIVATE hipblaslt-fp64-emulation)
