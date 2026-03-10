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
    target.addIllegalDialect<PolyDialect>();// 이 패스가 끝난 직후에 poly 연산이 단 하나라도 남아있으면 그것은 문법적 오류(Illegal)로 간주

    RewritePatternSet patterns(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace poly
}  // namespace tutorial
}  // namespace mlir