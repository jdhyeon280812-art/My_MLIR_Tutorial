#ifndef LIB_TRANSFORM_AFFINE_PASSES_H_
#define LIB_TRANSFORM_AFFINE_PASSES_H_

#pragma once
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {
namespace tutorial {


// 1. 우리가 cpp 파일에 구현한 팩토리 함수들을 직접 선언해 줍니다.
std::unique_ptr<mlir::Pass> createAffineFullUnroll();
std::unique_ptr<mlir::Pass> createAffineFullUnrollPatternRewrite();

// 2. TableGen이 생성한 패스 등록(Registration) 함수만 쏙 빼옵니다.
#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_TRANSFORM_AFFINE_PASSES_H_