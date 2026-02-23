#include "Transform/Affine/Passes.h"
#include "Dialect/Poly/PolyDialect.h"
#include "Transform/Arith/MulToAdd.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"


int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tutorial::poly::PolyDialect>();
  mlir::registerAllDialects(registry);  // 모든 built-in MLIR dialect(affine, arith, func, llvm, ...) 등록

 // ✅ 새로운 방식 (TableGen이 Passes.h.inc 안에 만들어둔 등록 함수 호출)
  mlir::tutorial::registerAffineFullUnroll();
  mlir::tutorial::registerAffineFullUnrollPatternRewrite();
  mlir::PassRegistration<mlir::tutorial::MulToAddPass>();
  
  // ✅ 패스 등록 (이 한 줄이면 Passes.h에 정의된 모든 패스가 한 번에 등록됩니다!)
    mlir::tutorial::registerAffineFullUnroll();
    mlir::tutorial::registerAffineFullUnrollPatternRewrite();

  // 메인 엔진(MlirOptMain) 실행 -> 파일 읽기, 파싱, 에러 처리, 결과 출력 등 컴파일러의 복잡한 뒷단 작업을 MLIR 프레임워크에게 위임. 우리는 패스만 만들어서 등록하면 됨
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}