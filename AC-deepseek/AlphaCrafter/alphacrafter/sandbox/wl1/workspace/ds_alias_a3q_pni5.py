import pathlib, json
# recent miner1 results
for f in ['scripts/miner1_cycle9_results.json','scripts/miner1_screen_results.json','scripts/miner1_rev5x_metrics.json']:
    try:
        d = json.load(open(f))
        print(f, '->', json.dumps(d)[:600])
        print('---')
    except Exception as e:
        print(f, 'ERR', e)
