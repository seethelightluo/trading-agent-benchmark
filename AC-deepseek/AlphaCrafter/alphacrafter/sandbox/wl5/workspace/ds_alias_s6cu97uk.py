import json
for f in ['semi_down_ratio_20.json','vol_of_vol20x60.json','mom_120d_skip5.json']:
    d = json.load(open(f'factors/{f}'))
    print(f, '| expr:', d['calculation']['expression'][:120])
    print('   desc:', d['calculation']['description'][:200])
    print()
