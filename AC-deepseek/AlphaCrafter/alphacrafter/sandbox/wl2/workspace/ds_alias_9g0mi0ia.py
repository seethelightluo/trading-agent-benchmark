import json
d = json.load(open('factors/factor_ensemble.json'))
def summarize(o, depth=0):
    if isinstance(o, dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)):
                print('  '*depth + str(k) + ': ' + (f'<list {len(v)}>' if isinstance(v,list) else '<dict>'))
                if isinstance(v,dict):
                    summarize(v, depth+1)
            else:
                s = str(v)
                print('  '*depth + f'{k}: {s[:120]}')
summarize(d)