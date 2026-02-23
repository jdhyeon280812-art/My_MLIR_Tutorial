include "mlir/Pass/PassBase.td"

def MulToAdd : Pass<"arith-mul-to-add"> {
  let summary = "Convert multiplications to additions";
  let description = [{
    이 패스는 Arith dialect의 곱셈(MulIOp)을 덧셈(AddIOp)이나 시프트(ShLIOp)로 변환합니다.
  }];
  
  // ✅ namespace 생략하고 이름만 적어주세요.
  let constructor = "createMulToAddPass()";
}