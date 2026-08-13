import json, os
print("=== factor_ensemble.json ===")
print(open('factor_ensemble.json').read())
print("\n=== factors dir ===")
for f in sorted(os.listdir('factors')):
    print(f, os.path.getsize(f'factors/{f}'))
print("\n=== evicted ===")
for f in sorted(os.listdir('factors/evicted')):
    print(f)
print("\n=== quarantine ===")
for f in sorted(os.listdir('factors/quarantine')):
    print(f)
print("\n=== rejected ===")
for f in sorted(os.listdir('factors/rejected')):
    print(f)
