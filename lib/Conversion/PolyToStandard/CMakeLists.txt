# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(PolyToStandard)

# MLIR 및 LLVM CMake 모듈 경로 추가
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)
include(AddLLVM)

# 중요: TableGen 및 C++ 컴파일러가 MLIR 헤더와 .td 파일을 찾을 수 있도록 경로 설정
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# --- 1. Convert Pass 생성 ---
set(LLVM_TARGET_DEFINITIONS PolyToStandard.td)
mlir_tablegen(PolyToStandard.h.inc -gen-pass-decls)
add_public_tablegen_target(PolyToStandardIncGen)

add_mlir_dialect_library(PolyToStandard
    PolyToStandard.cpp
    DEPENDS
    PolyDialectIncGen
    PolyTypesIncGen
    PolyOpsIncGen
    LINK_LIBS PUBLIC
    Poly
    MLIRIR
    MLIRArithDialect
    MLIRTensorDialect
    MLIRTransforms
    MLIRFuncDialect
    MLIRFuncTransforms
)

# 생성된 .inc 파일과 현재 소스 디렉토리를 Include 경로에 추가
target_include_directories(PolyToStandard PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)