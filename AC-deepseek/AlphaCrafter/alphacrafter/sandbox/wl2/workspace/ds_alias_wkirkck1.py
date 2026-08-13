import json, os
# check latest miner_1 results jsons
fs = sorted([f for f in os.listdir('scripts') if 'miner_1' in f and f.endswith('.json')])
print('\n'.join(fs[-15:]))
