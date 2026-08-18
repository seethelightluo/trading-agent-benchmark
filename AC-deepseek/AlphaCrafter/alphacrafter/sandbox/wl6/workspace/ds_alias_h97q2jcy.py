import json
d = json.load(open('factors/mom_120d_skip5.json'))
sa = d['validation']['signal_artifact']
print('keys:', list(sa.keys()))
print('format:', sa.get('format'))
print('description:', sa.get('description'))
print('columns:', sa.get('columns'))
print('shape:', sa.get('shape'))
print('n_valid_values:', sa.get('n_valid_values'))
print('sha256:', sa.get('sha256'))
print('data[:80]:', sa.get('data','')[:80])
print()
print('benchmark_admission:', json.dumps(d.get('benchmark_admission', {}), indent=1)[:1200])