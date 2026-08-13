import json
d = json.load(open('scripts/miner_1_20290628_batch1_results.json'))
for k,v in d.items():
    m = v.get('metrics', v)
    print(k, '->', {kk: m.get(kk) for kk in ['ic','icir','ic_hit_ratio','n_ic_dates','coverage_asset_days','turnover_10_rank','max_abs_library_correlation','decay_ic_by_horizon']})
    print('  pass:', v.get('pass'))
