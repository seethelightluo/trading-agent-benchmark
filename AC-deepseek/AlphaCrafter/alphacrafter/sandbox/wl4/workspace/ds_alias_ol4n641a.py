
import glob, os
# check for any miner output files
for f in sorted(glob.glob('scripts/miner_3*')):
    print(f, os.path.getsize(f))
print('---')
# check recent py files modified
for f in sorted(glob.glob('*.py')):
    print(f, os.path.getsize(f))
