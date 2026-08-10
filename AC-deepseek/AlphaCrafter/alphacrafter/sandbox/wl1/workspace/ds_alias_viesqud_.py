import os, json
fs = sorted(f for f in os.listdir('factors') if not f.endswith('.bak') and not f.endswith('.npy'))
print('non-bak factor files:', len(fs))
# read all to get status + factor_id
rows=[]
for f in fs:
    try:
        d=json.load(open('factors/'+f))
        rows.append((f, d.get('factor_id'), d.get('validation',{}).get('status'), d.get('last_validated','')))
    except Exception as e:
        rows.append((f,'ERR',str(e),''))
for r in rows: print(r)
