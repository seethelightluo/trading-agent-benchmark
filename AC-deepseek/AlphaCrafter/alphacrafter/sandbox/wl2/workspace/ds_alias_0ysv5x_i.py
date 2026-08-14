import json
d = json.load(open('factors/factor_ensemble.json'))
print("SELECTED FACTORS:")
for f in d['selected_factors']:
    print(json.dumps(f, indent=1)[:600])
print("\nNOTABLE OBSERVATIONS:")
for n in d.get('notable_observations', []):
    print('-', str(n)[:400])
print("\nMARKET ASSESSMENT keys:", list(d['market_assessment'].keys()))