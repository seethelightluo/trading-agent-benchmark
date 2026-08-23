import json
with open('factors/beta_VIX_60.json') as f:
    d = json.load(f)
# Just print keys and structure (not the whole artifact)
def summary(obj, depth=0, maxd=2):
    if depth > maxd: return '...'
    if isinstance(obj, dict):
        return {k: summary(v, depth+1, maxd) for k, v in list(obj.items())[:8]}
    if isinstance(obj, list):
        return f"list[{len(obj)}]"
    if isinstance(obj, str) and len(obj) > 80:
        return obj[:80]+'...'
    return obj
print(json.dumps(summary(d), indent=1)[:2500])