import os
# find recent miner scripts
for f in sorted(os.listdir('scripts')):
    if 'miner' in f:
        print(f)
