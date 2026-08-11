
import json
d = json.load(open('factors/tail_ratio_20.json'))
# Print structure keys
def keys(x, prefix=''):
    if isinstance(x, dict):
        for k,v in x.items():
            if isinstance(v, dict):
                print(prefix+k+':')
                keys(v, prefix+'  ')
            else:
                s = str(v)
                print(prefix+k+':', s[:120])
    else:
        print(prefix, type(x).__name__, str(x)[:120])
keys(d)
