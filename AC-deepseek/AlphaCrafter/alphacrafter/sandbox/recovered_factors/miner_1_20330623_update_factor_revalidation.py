import json
p='factors/miner_1_20330303_signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d.json'
d=json.load(open(p,encoding='utf8'))
d['version']='revalidated_2033-06-23'
d['last_validated']='2033-06-23T09:30:00'
d['validation_timestamp']='2033-06-23T09:30:00'
v=d['validation']; v['period']='2020-01-01 through 2033-06-22; visibility-safe completed-close cutoff'; v['timestamp']='2033-06-23T09:30:00'; v['status']='EFFECTIVE'
v['metrics']={'primary_horizon_days':20,'daily_paper_ic':0.055381,'daily_paper_icir':0.160749,'ic_std':0.344518,'ic_standard_error':0.008314,'ic_hit_ratio':0.564356,'ic_dates':1717,'universe_instruments':15,'mean_valid_instruments_per_ic_date':10.011648,'signal_cell_coverage':0.351263,'valid_cells':22119,'mean_rank_turnover':0.211477,'turnover_dates':1736,'library_factors_screened':34,'max_abs_library_correlation':0.401579,'max_abs_library_correlation_factor':'miner_3_residual_upside_volume_confirmation_60d','max_abs_library_correlation_common_cells':15158,'decay':{'1d':{'ic':0.015591,'icir':0.047324,'hit_ratio':0.509217,'ic_dates':1736},'5d':{'ic':0.033712,'icir':0.104899,'hit_ratio':0.54157,'ic_dates':1732},'10d':{'ic':0.040983,'icir':0.124911,'hit_ratio':0.544296,'ic_dates':1727},'20d':{'ic':0.055381,'icir':0.160749,'hit_ratio':0.564356,'ic_dates':1717}}}
v['regime_notes']={'2020_2024_10d':{'ic_dates':0,'ic':None,'icir':None,'hit_ratio':None},'2025_2026_10d':{'ic_dates':48,'ic':0.009275,'icir':0.029474,'hit_ratio':0.541667},'2027_onward_10d':{'ic_dates':1679,'ic':0.041889,'icir':0.12751,'hit_ratio':0.544372},'interpretation':'Revalidation passes binding gates at 5, 10, and 20 days, with strongest evidence at 20 days. Independence remains below the 0.5000 ceiling. The 2027-onward period remains positive; the short 2025-26 period is weak. Mean IC cross-section is 10.01 assets, above the eight-instrument minimum.'}
s=d['benchmark_admission']['selected_metrics'];s.update({'ic':0.055381,'icir':0.160749,'max_abs_library_correlation':0.401579,'quality':0.008902449869})
json.dump(d,open(p,'w',encoding='utf8'),indent=2,ensure_ascii=False)
