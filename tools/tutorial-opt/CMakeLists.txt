add_executable(tutorial-opt
  tutorial-opt.cpp
)

target_link_libraries(tutorial-opt
  PRIVATE
  TutorialAffineTransforms # 우리가 만든 라이브러리 연결
  MulToAdd #MulToAddPass의 구현체가 담긴 라이브러리
  Poly # //lib/Dialect/Poly
  MLIRMlirOptMain
  MLIRIR
  MLIRAffineDialect
  MLIRFuncDialect
)