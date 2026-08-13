import json, os
# look at most recent miner_1 explore results
for f in ['scripts/miner_1_20290906_explore_batch.py']:
    pass
# read json results
for f in sorted([x for x in os.listdir('scripts') if 'miner' in x and x.endswith('.json')])[-6:]:
    try:
        d = json.load(open('scripts/'+f))
        print(f, '->', str(d)[:600])
        print('---')
    except Exception as e:
        print(f, 'ERR', e)
