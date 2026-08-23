import json, os

fs = sorted([f for f in os.listdir('factors') if f.endswith('.json') and '.bak' not in f])
for f in fs:
    try:
        d = json.load(open('factors/' + f))
        v = d.get('validation', {})
        m = v.get('metrics', {})
        print(f"{f:32s} id={str(d.get('factor_id')):22s} status={str(v.get('status')):12s} last={str(v.get('last_validated')):12s} ic={m.get('ic')} icir={m.get('icir')}")
    except Exception as e:
        print(f, 'ERR', repr(e))