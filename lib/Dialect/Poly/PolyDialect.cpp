#include "PolyDialect.h"

#include "PolyTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PolyDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "PolyTypes.cpp.inc"

namespace mlir {
namespace tutorial {
namespace poly {

void PolyDialect::initialize() {
  addTypes<
  #define GET_TYPEDEF_LIST
  #include "PolyTypes.cpp.inc"
      >();
}

}  // namespace poly
}  // namespace tutorial
}  // namespace mlir