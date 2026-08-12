import glob, os
for f in sorted(glob.glob('scripts/miner_3_2028*')):
    print(f, os.path.getsize(f))
# check results json for 2028
for f in sorted(glob.glob('scripts/*2028*results*.json')):
    print('RESULT:', f, os.path.getsize(f))
