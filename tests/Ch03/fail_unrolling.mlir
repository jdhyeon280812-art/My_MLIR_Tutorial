func.func @loop_fail(%N: index) {
  // 0부터 %N까지 반복 (N이 뭔지 몰라서 Unroll 불가능)
  affine.for %i = 0 to %N {
    %c0 = arith.constant 0 : i32
  }
  return
}