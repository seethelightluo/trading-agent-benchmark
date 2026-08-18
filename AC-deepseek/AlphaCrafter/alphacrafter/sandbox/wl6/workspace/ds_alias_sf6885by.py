import os
print('quarantine:', os.listdir('factors/quarantine'))
print('evicted:', os.listdir('factors/evicted'))
# What did miner_3 persist most recently?
import glob
for f in sorted(glob.glob('scripts/miner_3*results*.json'))[-3:]:
    print('==', f)
print(open('scripts/miner_3_20270909_screen_batch.py').read()[-2500:])