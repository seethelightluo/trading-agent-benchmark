import json
root = json.load(open('factor_ensemble.json'))
fac = json.load(open('factors/factor_ensemble.json'))
print("ROOT:", json.dumps(root))
print("FACTORS/:", json.dumps(fac))
print("SYNC" if root == fac else "DIFF")
