import os
# Find miner scripts
ms = [f for f in os.listdir('scripts') if 'miner' in f]
print("miner scripts:", len(ms))
for f in sorted(ms)[-15:]:
    print(f)
print()
print("=== shared warmup seed keys ===")
import json
seed = json.load(open('../persistent/shared_warmup_seed.json'))
if isinstance(seed, dict):
    print(list(seed.keys())[:20])
else:
    print(type(seed), len(seed))