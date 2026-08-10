import json
d = json.load(open('factors/bollinger_z_20d.json'))
g = d['signal_artifact_grid']
print(type(g), list(g.items())[:2] if isinstance(g, dict) else g[:2])
print('shape:', d['signal_artifact_shape'], 'fmt:', d['signal_artifact_format'])
d2 = json.load(open('factors/dxy_beta_cond_60x20.json'))
print('dxy keys:', d2.get('signal_artifact'), d2.get('signal_artifact_format'))
print('benchmark_admission:', json.dumps(d.get('benchmark_admission'))[:300])