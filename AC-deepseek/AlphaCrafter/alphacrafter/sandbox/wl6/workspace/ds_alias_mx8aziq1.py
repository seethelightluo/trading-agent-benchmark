import os, glob
# Check for any miner_2 scripts and recent miner_3 output json
for f in sorted(glob.glob('scripts/*2027*results*.json')) + sorted(glob.glob('scripts/miner_2*')):
    print(f, os.path.getsize(f) if os.path.exists(f) else '')
# check trader probe target
print(open('scripts/trader_20270923_probe_target.py').read()[:3000])