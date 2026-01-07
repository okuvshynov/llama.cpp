function(ggml_add_cpu_backend_features cpu_name arch)
    # The feature detection code is compiled as a separate target so that
    # it can be built without the architecture flags
    # Since multiple variants of the CPU backend may be included in the same
    # build, using set_source_files_properties() to set the arch flags is not possible
    set(GGML_CPU_FEATS_NAME ${cpu_name}-feats)
    add_library(${GGML_CPU_FEATS_NAME} OBJECT ggml-cpu/arch/${arch}/cpu-feats.cpp)
    target_include_directories(${GGML_CPU_FEATS_NAME} PRIVATE . ../include)
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE ${ARGN})
    target_compile_definitions(${GGML_CPU_FEATS_NAME} PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED)
    set_target_properties(${GGML_CPU_FEATS_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${cpu_name} PRIVATE ${GGML_CPU_FEATS_NAME})
endfunction()

function(ggml_add_cpu_backend_variant_impl tag_name)
    if (tag_name)
        set(GGML_CPU_NAME ggml-cpu-${tag_name})
    else()
        set(GGML_CPU_NAME ggml-cpu)
    endif()

    ggml_add_backend_library(${GGML_CPU_NAME})

    list (APPEND GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/repack.cpp
        ggml-cpu/repack.h
        ggml-cpu/hbm.cpp
        ggml-cpu/hbm.h
        ggml-cpu/quants.c
        ggml-cpu/quants.h
        ggml-cpu/traits.cpp
        ggml-cpu/traits.h
        ggml-cpu/ggml-cpu-impl.h
        ggml-cpu/common.h
        ggml-cpu/binary-ops.h
        ggml-cpu/binary-ops.cpp
        ggml-cpu/unary-ops.h
        ggml-cpu/unary-ops.cpp
        ggml-cpu/simd-mappings.h
        ggml-cpu/vec.h
        ggml-cpu/vec.cpp
        ggml-cpu/ops.h
        ggml-cpu/ops.cpp
        )

    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)
    target_include_directories(${GGML_CPU_NAME} PRIVATE . ggml-cpu)

    if (APPLE AND GGML_ACCELERATE)
        find_library(ACCELERATE_FRAMEWORK Accelerate)
        if (ACCELERATE_FRAMEWORK)
            message(STATUS "Accelerate framework found")

            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_ACCELERATE)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_NEW_LAPACK)
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE ACCELERATE_LAPACK_ILP64)

            target_link_libraries(${GGML_CPU_NAME} PRIVATE ${ACCELERATE_FRAMEWORK})
        else()
            message(WARNING "Accelerate framework not found")
        endif()
    endif()

    if (GGML_OPENMP)
        find_package(OpenMP)
        if (OpenMP_FOUND)
            set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
            target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_OPENMP)

            target_link_libraries(${GGML_CPU_NAME} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
        else()
            set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
            message(WARNING "OpenMP not found")
        endif()
    endif()


    if (GGML_CPU_HBM)
        find_library(memkind memkind REQUIRED)

        message(STATUS "Using memkind for CPU HBM")

        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_HBM)

        target_link_libraries(${GGML_CPU_NAME} PUBLIC memkind)
    endif()

    if (GGML_SYSTEM_ARCH STREQUAL "ARM")
        message(STATUS "ARM detected")
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/arch/arm/quants.c
            ggml-cpu/arch/arm/repack.cpp
            )

        if (MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
            message(FATAL_ERROR "MSVC is not supported for ARM, use clang")
        else()
            check_cxx_compiler_flag(-mfp16-format=ieee GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E)
            if (NOT "${GGML_COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
                list(APPEND ARCH_FLAGS -mfp16-format=ieee)
            endif()

            if (GGML_NATIVE)
                # -mcpu=native does not always enable all the features in some compilers,
                # so we check for them manually and enable them if available

                execute_process(
                    COMMAND ${CMAKE_C_COMPILER} -mcpu=native -E -v -
                    INPUT_FILE "/dev/null"
                    OUTPUT_QUIET
                    ERROR_VARIABLE ARM_MCPU
                    RESULT_VARIABLE ARM_MCPU_RESULT
                )
                if (NOT ARM_MCPU_RESULT)
                    string(REGEX MATCH "-mcpu=[^ ']+" ARM_MCPU_FLAG "${ARM_MCPU}")
                    string(REGEX MATCH "-march=[^ ']+" ARM_MARCH_FLAG "${ARM_MCPU}")

                    # on some old GCC we need to read -march=
                    if (ARM_MARCH_FLAG AND NOT "${ARM_MARCH_FLAG}" STREQUAL "-march=native")
                        set(ARM_NATIVE_FLAG "${ARM_MARCH_FLAG}")
                    elseif(ARM_MCPU_FLAG AND NOT "${ARM_MCPU_FLAG}" STREQUAL "-mcpu=native")
                        set(ARM_NATIVE_FLAG "${ARM_MCPU_FLAG}")
                    endif()
                endif()

                if ("${ARM_NATIVE_FLAG}" STREQUAL "")
                    set(ARM_NATIVE_FLAG -mcpu=native)
                    message(WARNING "ARM -march/-mcpu not found, -mcpu=native will be used")
                else()
                    message(STATUS "ARM detected flags: ${ARM_NATIVE_FLAG}")
                endif()

                include(CheckCXXSourceRuns)

                macro(check_arm_feature tag feature code)
                    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
                    set(CMAKE_REQUIRED_FLAGS "${ARM_NATIVE_FLAG}+${tag}")
                    check_cxx_source_runs("${code}" GGML_MACHINE_SUPPORTS_${tag})
                    if (GGML_MACHINE_SUPPORTS_${tag})
                        set(ARM_NATIVE_FLAG_FIX "${ARM_NATIVE_FLAG_FIX}+${tag}")
                    else()
                        set(CMAKE_REQUIRED_FLAGS "${ARM_NATIVE_FLAG}+no${tag}")
                        check_cxx_source_compiles("int main() { return 0; }" GGML_MACHINE_SUPPORTS_no${tag})
                        if (GGML_MACHINE_SUPPORTS_no${tag})
                            set(ARM_NATIVE_FLAG_FIX "${ARM_NATIVE_FLAG_FIX}+no${tag}")
                            list(APPEND ARCH_FLAGS -U__ARM_FEATURE_${feature})
                        endif()
                    endif()
                    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
                endmacro()

                check_arm_feature(dotprod DOTPROD     "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }")
                check_arm_feature(i8mm    MATMUL_INT8 "#include <arm_neon.h>\nint main() { int8x16_t _a, _b; volatile int32x4_t _s = vmmlaq_s32(_s, _a, _b); return 0; }")
                check_arm_feature(sve     SVE         "#include <arm_sve.h>\nint main()  { svfloat32_t _a, _b; volatile svfloat32_t _c = svadd_f32_z(svptrue_b8(), _a, _b); return 0; }")
                check_arm_feature(sme     SME         "#include <arm_sme.h>\n__arm_locally_streaming int main() { __asm__ volatile(\"smstart; smstop;\"); return 0; }")

                list(APPEND ARCH_FLAGS "${ARM_NATIVE_FLAG}${ARM_NATIVE_FLAG_FIX}")
            else()
                if (GGML_CPU_ARM_ARCH)
                    list(APPEND ARCH_FLAGS -march=${GGML_CPU_ARM_ARCH})
                elseif(GGML_CPU_ALL_VARIANTS)
                    # Begin with the lowest baseline
                    set(ARM_MCPU "armv8-a")
                    set(ARCH_TAGS "")
                    set(ARCH_DEFINITIONS "")

                    # When a feature is selected, bump the MCPU to the first
                    # version that supported it
                    if (GGML_INTERNAL_DOTPROD)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+dotprod")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_DOTPROD)
                    endif()
                    if (GGML_INTERNAL_FP16_VECTOR_ARITHMETIC)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+fp16")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_FP16_VECTOR_ARITHMETIC)
                    endif()
                    if (GGML_INTERNAL_SVE)
                        set(ARM_MCPU "armv8.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sve")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SVE)
                    endif()
                    if (GGML_INTERNAL_MATMUL_INT8)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+i8mm")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_MATMUL_INT8)
                    endif()
                    if (GGML_INTERNAL_SVE2)
                        set(ARM_MCPU "armv8.6-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sve2")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SVE2)
                    endif()
                    if (GGML_INTERNAL_NOSVE)
                        set(ARCH_TAGS "${ARCH_TAGS}+nosve")
                    endif()
                    if (GGML_INTERNAL_SME)
                        set(ARM_MCPU "armv9.2-a")
                        set(ARCH_TAGS "${ARCH_TAGS}+sme")
                        list(APPEND ARCH_DEFINITIONS GGML_USE_SME)
                    endif()
                    list(APPEND ARCH_FLAGS "-march=${ARM_MCPU}${ARCH_TAGS}")
                    ggml_add_cpu_backend_features(${GGML_CPU_NAME} arm ${ARCH_DEFINITIONS})
                endif()
            endif()

            message(STATUS "Checking for ARM features using flags:")
            foreach(flag IN LISTS ARCH_FLAGS)
                message(STATUS "  ${flag}")
            endforeach()

            include(CheckCXXSourceCompiles)
            set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
            string(REPLACE ";" " " ARCH_FLAGS_STR "${ARCH_FLAGS}")
            set(CMAKE_REQUIRED_FLAGS "${ARCH_FLAGS_STR}")
            foreach(feature DOTPROD SVE MATMUL_INT8 FMA FP16_VECTOR_ARITHMETIC SME)
                set(ARM_FEATURE "HAVE_${feature}")
                check_cxx_source_compiles(
                    "
                    #if !defined(__ARM_FEATURE_${feature})
                    #  error \"Feature ${feature} is not defined\"
                    #endif
                    int main() { return 0; }
                    "
                    ${ARM_FEATURE}
                )
            endforeach()
            set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})
        endif()
    else()
        message(WARNING "Unknown CPU architecture (not ARM). Falling back to generic implementations.")
        list(APPEND ARCH_FLAGS -DGGML_CPU_GENERIC)
    endif()

    if (GGML_CPU_REPACK)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_USE_CPU_REPACK)
    endif()

    if (GGML_CPU_KLEIDIAI)
        message(STATUS "Using KleidiAI optimized kernels if applicable")

        # Disable the KleidiAI tests
        set(KLEIDIAI_BUILD_TESTS  OFF)

        # Fetch KleidiAI sources:
        include(FetchContent)
        set(KLEIDIAI_COMMIT_TAG "v1.16.0")
        set(KLEIDIAI_DOWNLOAD_URL "https://github.com/ARM-software/kleidiai/archive/refs/tags/${KLEIDIAI_COMMIT_TAG}.tar.gz")
        set(KLEIDIAI_ARCHIVE_MD5  "0a9e9008adb6031f9e8cf70dff4a3321")

        if (POLICY CMP0135)
            cmake_policy(SET CMP0135 NEW)
        endif()

        FetchContent_Declare(KleidiAI_Download
            URL ${KLEIDIAI_DOWNLOAD_URL}
            DOWNLOAD_EXTRACT_TIMESTAMP NEW
            URL_HASH MD5=${KLEIDIAI_ARCHIVE_MD5})

        FetchContent_MakeAvailable(KleidiAI_Download)
        FetchContent_GetProperties(KleidiAI_Download
            SOURCE_DIR  KLEIDIAI_SRC
            POPULATED   KLEIDIAI_POPULATED)

        if (NOT KLEIDIAI_POPULATED)
            message(FATAL_ERROR "KleidiAI source downloaded failed.")
        endif()

        add_compile_definitions(GGML_USE_CPU_KLEIDIAI)

        # Remove kleidiai target after fetching it
        if (TARGET kleidiai)
            set_target_properties(kleidiai PROPERTIES EXCLUDE_FROM_ALL TRUE)
        endif()

        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/kleidiai/kleidiai.cpp
            ggml-cpu/kleidiai/kernels.cpp
            ggml-cpu/kleidiai/kleidiai.h
            ggml-cpu/kleidiai/kernels.h
            )

        # KleidiAI
        include_directories(
            ${KLEIDIAI_SRC}/
            ${KLEIDIAI_SRC}/kai/
            ${KLEIDIAI_SRC}/kai/ukernels/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/)

        set(ARCH_FLAGS_TEMP "${ARCH_FLAGS}")
        if (NOT ARCH_FLAGS_TEMP)
            string(REGEX MATCH "-march=[^ ]+" ARCH_FLAGS_TEMP "${CMAKE_C_FLAGS}")
        endif()
        string(FIND "${ARCH_FLAGS_TEMP}" "+dotprod" DOTPROD_ENABLED)
        string(FIND "${ARCH_FLAGS_TEMP}" "+i8mm" I8MM_ENABLED)
        string(FIND "${ARCH_FLAGS_TEMP}" "+sme" SME_ENABLED)
        string(FIND "${ARCH_FLAGS_TEMP}" "+sve" SVE_ENABLED)

        set(PRIVATE_ARCH_FLAGS ${ARCH_FLAGS_TEMP})

        list(APPEND GGML_KLEIDIAI_SOURCES
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p4x8sb_f32_neon.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32ps1s0scalef16_qsu4c32s16s0_neon.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32_neon.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qai8dxp_f32.c
            ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c)

        if (NOT DOTPROD_ENABLED MATCHES -1)
            list(APPEND GGML_KLEIDIAI_SOURCES
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod.c)
        endif()

        if (NOT I8MM_ENABLED MATCHES -1)
            list(APPEND GGML_KLEIDIAI_SOURCES
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.c)
        endif()

        if (NOT SME_ENABLED MATCHES -1)
            list(APPEND GGML_KLEIDIAI_SOURCES
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme2_mopa.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme2_sdot.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme2_mopa_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.c
                ${KLEIDIAI_SRC}/kai/kai_common_sme_asm.S)
            set(PRIVATE_ARCH_FLAGS "-fno-tree-vectorize;${PRIVATE_ARCH_FLAGS}+sve+sve2")
        endif()

        if (NOT SVE_ENABLED MATCHES -1)
            list(APPEND GGML_KLEIDIAI_SOURCES
                ${KLEIDIAI_SRC}/kai/kai_common_sve_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.c
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm_asm.S
                ${KLEIDIAI_SRC}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.c)
        endif()

        set_source_files_properties(${GGML_KLEIDIAI_SOURCES} PROPERTIES COMPILE_OPTIONS "${PRIVATE_ARCH_FLAGS}")
        list(APPEND GGML_CPU_SOURCES ${GGML_KLEIDIAI_SOURCES})
    endif()

    message(STATUS "Adding CPU backend variant ${GGML_CPU_NAME}: ${ARCH_FLAGS} ${ARCH_DEFINITIONS}")
    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})
    target_compile_options(${GGML_CPU_NAME} PRIVATE ${ARCH_FLAGS})
    target_compile_definitions(${GGML_CPU_NAME} PRIVATE ${ARCH_DEFINITIONS})

    if (EMSCRIPTEN)
        set_target_properties(${GGML_CPU_NAME} PROPERTIES COMPILE_FLAGS "-msimd128")
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        # The compiler automatically enables "-ffast-math" which can cause NaNs in tests due to "-fassociative-math"
        target_compile_options(${GGML_CPU_NAME} PRIVATE "-fno-associative-math")
    endif()
endfunction()
