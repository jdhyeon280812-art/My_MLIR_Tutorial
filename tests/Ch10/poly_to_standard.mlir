// RUN: tutorial-opt --poly-to-standard %s | FileCheck %s

// CHECK-LABEL: test_lower_add
func.func @test_lower_add(%0 : !poly.poly<10>, %1 : !poly.poly<10>) -> !poly.poly<10> {
  // CHECK: arith.addi
  %2 = poly.add %0, %1: !poly.poly<10>
  return %2 : !poly.poly<10>
}

// CHECK-LABEL: test_lower_add_and_fold
func.func @test_lower_add_and_fold() {
  // CHECK: arith.constant dense<[2, 3, 4]> : tensor<3xi32>
  %0 = poly.constant dense<[2, 3, 4]> : tensor<3xi32> : !poly.poly<10>
  // CHECK: arith.constant dense<[3, 4, 5]> : tensor<3xi32>
  %1 = poly.constant dense<[3, 4, 5]> : tensor<3xi32> : !poly.poly<10>
  // would be an addi, but it was folded
  // CHECK: arith.constant
  %2 = poly.add %0, %1: !poly.poly<10>
  return
}