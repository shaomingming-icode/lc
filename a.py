import os
import subprocess
import sys

result = subprocess.run(["clang-tidy", "--warnings-as-errors=*", "-p", "build", "c.cpp"], capture_output=True, text=True)
print("clang_tidy errors:")
print(result.stdout)
print(result.stderr)
sys.exit(result.returncode)