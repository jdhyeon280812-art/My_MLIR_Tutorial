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
    // TODO: implement pass
  }
};

}  // namespace poly
}  // namespace tutorial
}  // namespace mlir