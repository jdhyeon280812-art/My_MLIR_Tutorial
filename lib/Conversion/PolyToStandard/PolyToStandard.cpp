#include "PolyToStandard.h"

#include "Dialect/Poly/PolyOps.h"
#include "Dialect/Poly/PolyTypes.h"
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace tutorial {
namespace poly {

#define GEN_PASS_DEF_POLYTOSTANDARD
#include "PolyToStandard.h.inc"

struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
  using PolyToStandardBase::PolyToStandardBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    
    ConversionTarget target(*context);
    target.addIllegalDialect<PolyDialect>();

    RewritePatternSet patterns(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace poly
}  // namespace tutorial
}  // namespace mlir