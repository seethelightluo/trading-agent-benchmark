import json
with open('factors/spx_corr60.json') as f:
    d = json.load(f)
sa = d.get('signal_artifact', {})
print("signal_artifact keys:", list(sa.keys()))
print("dates len:", len(sa.get('dates', [])), "values len:", len(sa.get('values', [])) if isinstance(sa.get('values'), list) else type(sa.get('values')))
print("sample dates:", sa.get('dates', [])[:3], "...", sa.get('dates', [])[-3:])
v = sa.get('values')
if isinstance(v, list):
    print("sample values:", v[:3], "...", v[-3:])
elif v is not None:
    print("values type:", type(v), "shape:", getattr(v, 'shape', None))