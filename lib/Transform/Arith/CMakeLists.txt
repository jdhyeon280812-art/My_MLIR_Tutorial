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