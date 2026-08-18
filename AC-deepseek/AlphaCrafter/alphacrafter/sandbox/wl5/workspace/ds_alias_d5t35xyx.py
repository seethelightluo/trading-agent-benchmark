import json
try:
    d = json.load(open('scripts/miner_2_20291004_revalidate_results.json')) if False else None
except Exception as e:
    print(e)
# check miner_1 recent results
try:
    d1 = json.load(open('scripts/miner_1_20291018_revalidate_results.json'))
    for k, v in d1.items():
        m = v.get('metrics', {})
        print(k, "IC=", round(m.get('ic', float('nan')),4), "ICIR=", round(m.get('icir', float('nan')),3), "n=", m.get('n_ic_dates'), "pass=", v.get('passed'))
except Exception as e:
    print("err", e)
