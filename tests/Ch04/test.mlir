func.func @unroll_me() {
  affine.for %i = 0 to 3 {
    "test.foo"(%i) : (index) -> ()
  }
  return
}