python -c "
import json
d = json.load(open('scripts/miner_3_20270923_revalidate_results.json'))
for k, v in d.items():
    print(f\"{k:30s} ic={v.get('ic',0):+.4f} icir={v.get('icir',0):+.4f} ok={v.get('ok')} n={v.get('n_ic_dates')}\")
"