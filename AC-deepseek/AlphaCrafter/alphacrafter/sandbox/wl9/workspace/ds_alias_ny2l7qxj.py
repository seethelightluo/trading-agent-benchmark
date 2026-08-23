import json
from pathlib import Path
# Check a factor file structure (truncated fields only)
d = json.load(open('factors/skew_20d.json'))
def summarize(o, depth=0, maxd=2, key=''):
    if depth>maxd: return
    if isinstance(o, dict):
        for k,v in list(o.items())[:14]:
            if k=='validation':
                print('  '*depth, k, '->', {kk:(str(vv)[:90]) for kk,vv in v.items()})
            elif isinstance(v,(dict,list)):
                print('  '*depth, k, type(v).__name__)
                summarize(v,depth+1,maxd,key=k)
            else:
                print('  '*depth, f'{k}: {str(v)[:110]}')
    elif isinstance(o,list):
        print('  '*depth, f'[list len {len(o)}]')
        if o: summarize(o[0],depth+1,maxd)
summarize(d)