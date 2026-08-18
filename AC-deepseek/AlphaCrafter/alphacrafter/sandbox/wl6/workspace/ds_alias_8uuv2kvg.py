import json
d = json.load(open('factors/mom_120d_skip5.json'))
# print structure keys and a compact view of metadata (not signal arrays)
def summary(obj, depth=0, maxdepth=2):
    if depth > maxdepth: return
    if isinstance(obj, dict):
        for k,v in obj.items():
            if isinstance(v, (dict, list)):
                if isinstance(v, list) and len(v) > 5:
                    print('  '*depth + f'{k}: list[{len(v)}]')
                else:
                    print('  '*depth + f'{k}:')
                    summary(v, depth+1, maxdepth)
            else:
                s = str(v)
                print('  '*depth + f'{k}: {s[:100]}')
summary(d)
print('TOP KEYS:', list(d.keys()))