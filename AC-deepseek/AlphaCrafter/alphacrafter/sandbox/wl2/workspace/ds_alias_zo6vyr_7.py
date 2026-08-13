import json
for f in ['scripts/miner_1_20290906_explore_batch.py', 'scripts/miner_1_20290726_batch_explore.py']:
    pass
# read json results
for f in ['scripts/_miner1_cycle31_explore_results.json','scripts/_miner1_cycle31b_explore_results.json']:
    try:
        d = json.load(open(f))
        print(f, '->', str(d)[:800])
    except Exception as e:
        print(f, 'ERR', e)
