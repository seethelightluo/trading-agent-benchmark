import json
d = json.load(open('factors/vol_adj_mom_accel_20x60.json'))
print(list(d.keys()))
for k,v in d.items():
    if k != 'signal': 
        s = str(v)
        print(k, ':', s[:500])
    else:
        print(k, ': signal array len', len(v))
