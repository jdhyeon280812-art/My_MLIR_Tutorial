#ifndef LIB_DIALECT_POLY_POLYPATTERNS_TD_
#define LIB_DIALECT_POLY_POLYPATTERNS_TD_

include "PolyOps.td"
include "mlir/Dialect/Complex/IR/ComplexOps.td"
include "mlir/IR/PatternBase.td"

def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>;

#endif  // LIB_DIALECT_POLY_POLYPATTERNS_TD_