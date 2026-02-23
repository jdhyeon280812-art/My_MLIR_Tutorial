#ifndef LIB_TRANSFORM_ARITH_PASSES_H_
#define LIB_TRANSFORM_ARITH_PASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {
namespace tutorial {

std::unique_ptr<mlir::Pass> createMulToAddPass();

// 1. TableGen이 만들어주는 패스 베이스 클래스 선언 (impl::MulToAddBase 등)
#define GEN_PASS_DECL
// 2. TableGen이 만들어주는 자동 등록 함수 (registerArithTutorialPasses)
#define GEN_PASS_REGISTRATION

#include "ArithPasses.h.inc"

} // namespace tutorial
} // namespace mlir

#endif // LIB_TRANSFORM_ARITH_PASSES_H_