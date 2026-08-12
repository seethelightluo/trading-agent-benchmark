import json, os, glob
files = glob.glob('scripts/miner3_*results.json') + glob.glob('scripts/miner3_*revalidate_results.json')
for f in sorted(files):
    print(os.path.getsize(f), f)
