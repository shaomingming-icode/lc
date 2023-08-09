import os
import subprocess
import sys

p = subprocess.run([ "clang-format", "-n", "--Werror", "c.cpp" ], capture_output=True)

print( 'exit status:', p.returncode )
print( 'stdout:', p.stdout.decode() )
print( 'stderr:', p.stderr.decode() )

sys.exit(p.returncode)