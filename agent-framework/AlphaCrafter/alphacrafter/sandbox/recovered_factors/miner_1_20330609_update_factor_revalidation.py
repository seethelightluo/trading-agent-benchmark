import json
p='factors/miner_1_20330303_signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d.json'
d=json.load(open(p,encoding='utf8'))
d['version']='revalidated_2033-06-09'
d['last_validated']='2033-06-09T09:30:00'
v=d['validation'];v['period']='2020-01-01 through 2033-06-08; visibility-safe completed-close cutoff';v['timestamp']='2033-06-09T09:30:00';v['status']='EFFECTIVE'
v['metrics']={'primary_horizon_days':20,'daily_paper_ic':0.055788,'daily_paper_icir':0.161872,'ic_std':0.344641,'ic_standard_error':0.008342,'ic_hit_ratio':0.565319,'ic_dates':1707,'universe_instruments':15,'mean_valid_instruments_per_ic_date':10.011716,'signal_cell_coverage':0.350509,'valid_cells':22019,'mean_rank_turnover':0.210989,'turnover_dates':1726,'library_factors_screened':34,'max_abs_library_correlation':0.401879,'max_abs_library_correlation_factor':'miner_3_residual_upside_volume_confirmation_60d','max_abs_library_correlation_common_cells':15098,'decay':{'1d':{'ic':0.015303,'icir':0.046509,'hit_ratio':0.508691,'ic_dates':1726},'5d':{'ic':0.033888,'icir':0.105441,'hit_ratio':0.541812,'ic_dates':1722},'10d':{'ic':0.041581,'icir':0.126521,'hit_ratio':0.545137,'ic_dates':1717},'20d':{'ic':0.055788,'icir':0.161872,'hit_ratio':0.565319,'ic_dates':1707}}}
v['regime_notes']={'2020_2024_10d':{'ic_dates':0,'ic':None,'icir':None,'hit_ratio':None},'2025_2026_10d':{'ic_dates':48,'ic':0.009275,'icir':0.029474,'hit_ratio':0.541667},'2027_onward_10d':{'ic_dates':1669,'ic':0.04251,'icir':0.129176,'hit_ratio':0.545237},'interpretation':'Revalidation passes binding gates at 5, 10, and 20 days, with strongest evidence at 20 days. Independence remains below the 0.5000 ceiling. The 2027-onward period remains positive; 2025-26 is short and weak. Mean IC cross-section is 10.01 assets, above the eight-instrument minimum.'}
json.dump(d,open(p,'w',encoding='utf8'),indent=2,ensure_ascii=False)
print('updated',p,d['last_validated'],v['metrics']['max_abs_library_correlation'])
