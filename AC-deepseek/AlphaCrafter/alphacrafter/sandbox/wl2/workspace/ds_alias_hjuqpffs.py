import json, os
# check revalidate results of miner_3 most recent
for f in ['scripts/miner_3_20291115_revalidate_library.py']:
    print(f, os.path.exists(f))
# search for any recent results json 2030
fs = [x for x in os.listdir('scripts') if x.endswith('.json') and ('2030' in x or '2029' in x)]
print('\n'.join(fs))
