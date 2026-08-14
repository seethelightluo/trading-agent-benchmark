import json
for fn in ['factors/calmness_20.json','factors/volcluster_60.json']:
    with open(fn) as f:
        d = json.load(f)
    print("====", fn)
    print("keys:", list(d.keys()))
    v = d.get('validation', {})
    print("status:", v.get('status'), "| last_validated:", v.get('last_validated'), "| period:", v.get('period'))
    m = v.get('metrics', {})
    print("metrics keys:", list(m.keys()))
    print("ic:", m.get('ic'), "icir:", m.get('icir'), "n_ic_dates:", m.get('n_ic_dates'))
    print("signal_artifact present:", 'signal_artifact' in d)
    if 'signal_artifact' in d:
        sa = d['signal_artifact']
        print("  sa keys:", list(sa.keys()), "dates:", len(sa.get('dates',[])), "vals:", len(sa.get('values',[])))