import json
for fn in ['factors/calmness_20.json','factors/volcluster_60.json','factors/downbeta_spx_60.json']:
    with open(fn) as f:
        d = json.load(f)
    print("====", fn)
    print("keys:", list(d.keys()))
    v = d.get('validation', {})
    print("status:", v.get('status'), "| last_validated:", v.get('last_validated'), "| period:", v.get('period'))
    m = v.get('metrics', {})
    print("ic:", m.get('ic'), "icir:", m.get('icir'), "n_ic_dates:", m.get('n_ic_dates'))
    sa = d.get('signal_artifact')
    print("signal_artifact type:", type(sa))
    if isinstance(sa, dict):
        print("  sa keys:", list(sa.keys()), "dates:", len(sa.get('dates',[])), "vals:", len(sa.get('values',[])))
    elif isinstance(sa, str):
        print("  sa (str):", sa[:200])