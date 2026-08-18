
import json
d = json.load(open('scripts/miner_3_20300207_revalidate_results.json'))
print("visible_through:", d['visible_through'], "n_dates:", d['n_dates'])
for fid, res in d['results'].items():
    print(f"{fid:25s} ic={res['ic']:8.4f} icir={res['icir']:8.4f} hit={res['ic_hit_ratio']:.3f} n={res['n_ic_dates']:5d} maxrho={res.get('max_abs_library_correlation')}")
    print("   regime_recent:", {k: v for k, v in res.get('regime_recent', {}).items()})
