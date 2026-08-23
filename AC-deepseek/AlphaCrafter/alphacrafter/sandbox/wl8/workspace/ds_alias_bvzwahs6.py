import json
try:
    d=json.load(open('factors/factor_ensemble.json'))
    print('ensemble keys:', list(d.keys()))
    print('asof:', d.get('asof'))
    print('selected:', json.dumps(d.get('selected_factors'), indent=2))
except Exception as e:
    print('err', e)
