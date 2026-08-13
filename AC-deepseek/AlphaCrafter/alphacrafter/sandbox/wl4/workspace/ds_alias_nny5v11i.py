import os
# Show last few miner_3 scripts and their sizes
for f in sorted(os.listdir('scripts')):
    if 'miner_3' in f:
        print(f, os.path.getsize(f'scripts/{f}'))
