#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Pass/Pass.h"
#include "Passes.h"

namespace mlir {
namespace tutorial {


#define GEN_PASS_DEF_AFFINEFULLUNROLL
#include "Passes.h.inc"

using mlir::affine::AffineForOp;
using mlir::affine::loopUnrollFull;

// A pass that manually walks the IR
struct AffineFullUnroll : impl::AffineFullUnrollBase<AffineFullUnroll> {
  using AffineFullUnrollBase<AffineFullUnroll>::AffineFullUnrollBase;

  void runOnOperation() override {
    getOperation()->walk([&](AffineForOp op) {
      if (failed(loopUnrollFull(op))) {
        op.emitError("unrolling failed");
        signalPassFailure();
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createAffineFullUnroll() {
  return std::make_unique<AffineFullUnroll>(); // 이름이 위에 정의된 struct 이름과 같아야 합니다.
}

} // namespace tutorial
} // namespace mlir