set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(ArithPasses.h.inc -gen-pass-decls -name Arith)
mlir_tablegen(ArithPasses.md -gen-pass-doc)
add_public_tablegen_target(ArithPassesIncGen)

add_mlir_library(MulToAdd
  MulToAdd.cpp

  # 헤더 파일은 필수는 아니지만 IDE 지원을 위해 명시할 수 있습니다.
  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
)

#add_dependencies(MulToAdd ArithPassIncGen)

# Include 경로 설정 (.inc 파일 인식용)
target_include_directories(MulToAdd PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)