import json, os
# current library effective statuses
for f in sorted(os.listdir('factors')):
    if f.endswith('.json') and not f.endswith('.bak'):
        try:
            d = json.load(open('factors/'+f))
            status = d.get('validation',{}).get('status','?')
            lv = d.get('validation',{}).get('last_validated', d.get('last_validated','?'))
            fid = d.get('factor_id', f)
            print(f"{fid:35s} {status:12s} lv={str(lv)[:10]}")
        except Exception as e:
            print(f, 'ERR', e)
