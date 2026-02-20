# 파일명: lit.cfg.py
import lit.formats

# 1. 테스트 스위트 이름 (아무거나 상관없음)
config.name = "My MLIR Practice"

# 2. 테스트 형식 지정
# ShTest(True)는 "파일 안에 있는 RUN 명령어를 쉘 스크립트처럼 실행하라"는 뜻입니다.
config.test_format = lit.formats.ShTest(True)

# 3. 테스트할 파일 확장자 지정
config.suffixes = ['.mlir']

tool_relpaths = [
    "llvm-project/mlir",
    "llvm-project/llvm",
    "mlir_tutorial/tools",
]
# 4. (선택사항) 만약 PATH 설정이 귀찮다면 여기서 강제로 지정할 수도 있습니다.
# import os
# config.environment['PATH'] = "/Users/jeongdonghyeon/llvm-project/build/bin:" + os.environ['PATH']

