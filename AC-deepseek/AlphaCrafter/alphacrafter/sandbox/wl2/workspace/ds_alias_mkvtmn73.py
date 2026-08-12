import json
d = json.load(open('scripts/miner_3_20270923_revalidate_results.json'))
for fid, s in d.items():
    print(f"{fid:28s} ic={s.get('ic'):+.4f} icir={s.get('icir'):+.4f} hit={s.get('hit')} turn={s.get('turnover_10d_rank')} cov={s.get('coverage')} maxrho={s.get('max_abs_library_correlation')} ok={s.get('ok')} lastval={s.get('last_validated')}")
