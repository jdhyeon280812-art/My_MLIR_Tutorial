#include "Transform/Arith/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

using arith::AddIOp;
using arith::ConstantOp;
using arith::MulIOp;

// Replace y = C*x with y = C/2*x + C/2*x, when C is a power of 2, otherwise do
// nothing.
struct PowerOfTwoExpand :
  public OpRewritePattern<MulIOp> {
  PowerOfTwoExpand(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {
        
        Value lhs = op.getOperand(0);
        // canonicalization patterns ensure the constant is on the right, if there is a constant
        // See https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules
        Value rhs = op.getOperand(1); // 여기서 Value는 SSA 값을 나타내는 타입.
        // rhs가 상수인지 확인
        auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
        if (!rhsDefiningOp) {
        return failure();
        }
        // 2의 거듭제곱인지 확인
        int64_t value = rhsDefiningOp.value();
        bool is_power_of_two = (value & (value - 1)) == 0;

        if (!is_power_of_two) {
        return failure();
        }
        /*
        // 1. 절반 값을 가진 새로운 상수 생성
        ConstantOp newConstant = rewriter.create<ConstantOp>(
            rhsDefiningOp.getLoc(), rewriter.getIntegerAttr(rhs.getType(), value / 2));
        // 2. 새로 만든 상수와의 곱셈을 생성
        MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
        // 3. 새로 만든 곱셈끼리 더함
        AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, newMul);

        rewriter.replaceOp(op, newAdd); // 기존의 곱셈(op)을 덧셈(newAdd)으로 바꿈
        rewriter.eraseOp(rhsDefiningOp); // 기존의 상수는 제거. 더 이상 필요없음
        */
        
        // 1. 몇 번 시프트 해야 하는지 계산 (log2(value))
        // llvm::Log2_64 함수 사용
        int64_t shiftAmount = llvm::Log2_64(value);

        // 2. 시프트 양을 상수로 만듦
        ConstantOp shiftConstant = rewriter.create<ConstantOp>(
            rhsDefiningOp.getLoc(), rewriter.getIntegerAttr(rhs.getType(), shiftAmount));

        // 3. Shift Left 연산 생성 (arith.shli)
        arith::ShLIOp newShift = rewriter.create<arith::ShLIOp>(op.getLoc(), lhs, shiftConstant);

        // 4. 교체
        rewriter.replaceOp(op, newShift);
        rewriter.eraseOp(rhsDefiningOp);
        

    return success();
  }
};

// Replace y = 9*x with y = 8*x + x
struct PeelFromMul :
  public OpRewritePattern<MulIOp> {
  PeelFromMul(mlir::MLIRContext *context)
      : OpRewritePattern<MulIOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(MulIOp op,
                                PatternRewriter &rewriter) const override {

    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
    if (!rhsDefiningOp) {
      return failure();
    }

    int64_t value = rhsDefiningOp.value();

    // We are guaranteed `value` is not a power of two, because the greedy
    // rewrite engine ensures the PowerOfTwoExpand pattern is run first, since
    // it has higher benefit.

    ConstantOp newConstant = rewriter.create<ConstantOp>(
        rhsDefiningOp.getLoc(), rewriter.getIntegerAttr(rhs.getType(), value - 1));
    MulIOp newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
    AddIOp newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, lhs);

    rewriter.replaceOp(op, newAdd);
    rewriter.eraseOp(rhsDefiningOp);

    return success();
  }
};

#define GEN_PASS_DEF_MULTOADD
#include "ArithPasses.h.inc"

// ✅ Passes.h(TableGen)에서 선언된 베이스 클래스를 여기서 바로 상속받아 구현합니다.
struct MulToAddPass : public impl::MulToAddBase<MulToAddPass> {
  using MulToAddBase::MulToAddBase;

  void runOnOperation() override {
      mlir::RewritePatternSet patterns(&getContext());
      patterns.add<PowerOfTwoExpand>(&getContext());
      patterns.add<PeelFromMul>(&getContext());
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

// 생성 함수 구현
std::unique_ptr<mlir::Pass> createMulToAddPass() {
  return std::make_unique<MulToAddPass>();
}

} // namespace tutorial
} // namespace mlir