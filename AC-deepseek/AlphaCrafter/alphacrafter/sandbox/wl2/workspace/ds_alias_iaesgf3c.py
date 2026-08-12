import glob, os
# what factor ideas have been explored recently by miner1/miner3?
for f in sorted(glob.glob('scripts/*2028*.py')):
    print(f, os.path.getsize(f))
