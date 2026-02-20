cmake_minimum_required(VERSION 3.20)
project(tutorial-opt)

# LLVM/MLIR 설치 경로를 찾아줍니다 (환경에 따라 경로 수정 필요할 수 있음)
find_package(MLIR REQUIRED CONFIG)

# MLIR과 LLVM의 헤더 파일 경로를 컴파일러에게 알려줍니다.
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# RTTI 비활성화 (LLVM 빌드 설정과 맞춤)
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# [핵심] lib 폴더를 헤더 경로로 포함 (저자의 합쳐진 구조를 위해 필수!)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

add_subdirectory(lib)
add_subdirectory(tools)