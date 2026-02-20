# 1. TableGen 컴파일을 위한 타겟 파일 지정
set(LLVM_TARGET_DEFINITIONS Passes.td)

# 2. Passes.h.inc 생성 규칙
mlir_tablegen(Passes.h.inc -gen-pass-decls -name=Affine)

# 3. 문서 파일 생성 규칙 (필요시)
mlir_tablegen(AffinePasses.md -gen-pass-doc)

# 4. TableGen 타겟 추가 (Bazel의 name = "pass_inc_gen" 역할)
add_public_tablegen_target(AffinePassIncGen)

add_library(TutorialAffineTransforms
  AffineFullUnroll.cpp
  AffineFullUnrollPatternRewrite.cpp
  Passes.h
)

target_link_libraries(TutorialAffineTransforms
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRAffineUtils
  MLIRTransforms
)

add_dependencies(TutorialAffineTransforms AffinePassIncGen)
# (TableGen이 생성한 .inc 파일이 있는 빌드 폴더를 헤더 탐색 경로에 추가)
target_include_directories(TutorialAffineTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR})