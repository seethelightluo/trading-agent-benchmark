python -c "
import json,glob
for fp in sorted(glob.glob('factors/*.json')):
    if '.bak' in fp or 'factor_ensemble' in fp: continue
    d=json.load(open(fp))
    m=d.get('validation',{}).get('metrics',{})
    print(d['factor_id'],'| dir',d.get('expected_direction'),'| IC',m.get('ic'),'| ICIR',m.get('icir'),'| qual',m.get('quality'),'| turn_rank',m.get('turnover_10d_rank'),'| hicorr',m.get('max_abs_library_correlation'))
"