// RUN: tutorial-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @test_poly_fold
func.func @test_poly_fold() -> !poly.poly<10> {
  // 원래의 다항식: 1 + 2x + 3x^2
  %0 = poly.constant dense<[1, 2, 3]> : tensor<3xi32> : <10>
  
  // (1 + 2x + 3x^2) * (1 + 2x + 3x^2)
  %1 = poly.mul %0, %0 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
  %2 = poly.mul %0, %0 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
  
  // 두 결과를 더함
  %3 = poly.add %1, %2 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
  
  return %3 : !poly.poly<10>
}

// 컴파일러가 최적화를 수행했다면, 덧셈과 곱셈 명령어는 모두 사라져야 합니다.
// CHECK-NOT: poly.mul
// CHECK-NOT: poly.add

// 대신, 미리 계산된 최종 상수 하나만 남아있어야 합니다.
// CHECK: %{{.*}} = poly.constant dense<[2, 8, 20, 24, 18]> : tensor<5xi32> : <10>