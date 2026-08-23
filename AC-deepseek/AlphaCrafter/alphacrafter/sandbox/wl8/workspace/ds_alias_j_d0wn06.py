import json, os
print("=== factor_ensemble.json ===")
with open('factor_ensemble.json') as f:
    print(f.read())
print("=== active json files in factors/ (excluding .bak and dirs) ===")
for fn in sorted(os.listdir('factors/')):
    p = os.path.join('factors', fn)
    if os.path.isfile(p) and not fn.endswith('.bak'):
        print(fn, os.path.getsize(p))
print("=== rejected ===")
for fn in sorted(os.listdir('factors/rejected/'))[:20]:
    print(fn)