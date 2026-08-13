import json,glob,os
# print max corr library id for all factors
for f in sorted(glob.glob('factors/*.json')):
    if 'ensemble' in f: continue
    d=json.load(open(f))
    v=d.get('validation',{}).get('metrics',{})
    print(f"{d['factor_id']:26s} maxcorr={v.get('max_abs_library_correlation',0):.3f} with={v.get('max_corr_library_id','?'):24s} turn={v.get('turnover_10d_rank',0):.2f}")
