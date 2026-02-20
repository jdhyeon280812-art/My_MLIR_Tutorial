#ifndef LIB_TRANSFORM_AFFINE_PASSES_TD_
#define LIB_TRANSFORM_AFFINE_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AffineFullUnroll : Pass<"affine-full-unroll"> {
  let summary = "Fully unroll all affine loops";
  let description = [{
    Fully unroll all affine loops.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
  let constructor = "mlir::tutorial::createAffineFullUnroll()";
}

def AffineFullUnrollPatternRewrite : Pass<"affine-full-unroll-rewrite"> {
  let summary = "Fully unroll all affine loops using the pattern rewrite engine";
  let description = [{
    Fully unroll all affine loops using the pattern rewrite engine.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
  // ✅ TableGen의 자동 생성을 막고 우리가 직접 만들겠다고 선언
  let constructor = "mlir::tutorial::createAffineFullUnrollPatternRewrite()";
}

#endif  // LIB_TRANSFORM_AFFINE_PASSES_TD_