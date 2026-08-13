import json
d=json.load(open('factors/vol_regime_switch_20x60.json'))
print('id:',d['factor_id'],'dir:',d.get('expected_direction'))
print('tags:',d.get('tags'))
v=d['validation']['metrics']
print('ic',v['ic'],'icir',v['icir'],'hit',v['ic_hit_ratio'],'turn',v['turnover_10d_rank'],'maxcorr',v['max_abs_library_correlation'],v.get('max_corr_library_id'))
d2=json.load(open('factors/comm_basket_beta_60.json'))
print()
print('comm_basket dir:',d2.get('expected_direction'),'tags:',d2.get('tags'))
