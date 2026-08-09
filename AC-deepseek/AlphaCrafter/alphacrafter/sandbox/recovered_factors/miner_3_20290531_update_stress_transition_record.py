import json
p='factors/miner_3_20290503_residual_vix_dxy_stress_transition_60obs.json'
d=json.load(open(p)); v=d['validation']; m=v['metrics']
v['period']='2020-01-01 through 2029-05-30; point-in-time daily observations only'; v['status']='EFFECTIVE'
m.update({'selected_horizon_days':20,'daily_paper_ic':0.0367176884,'daily_paper_icir':0.1254153221,'hit_ratio':0.5264367816,'ic_dates':435,'mean_valid_instruments_per_ic_date':11.9908046,'coverage':0.1110686212,'valid_date_asset_cells':5228,'possible_date_asset_cells':47070,'turnover_10d_rank':0.6480024041,'decay':{'1d':{'ic':0.0046123601,'icir':0.016008516,'dates':436},'5d':{'ic':0.0041617404,'icir':0.0138935881,'dates':436},'10d':{'ic':0.0108595533,'icir':0.0370476805,'dates':436},'20d':{'ic':0.0367176884,'icir':0.1254153221,'dates':435}},'max_abs_library_correlation':0.1610250995,'closest_library_factor':'orthogonal_trend_acceleration_20_60obs','closest_common_date_asset_cells':5164,'quality_score_abs_ic_times_abs_icir':0.004605})
v['regime_notes']={'2026':{'ic':0.0547476333,'icir':0.1883229855,'dates':70},'2027':{'ic':0.1079191079,'icir':0.4523699531,'dates':37},'2028':{'ic':0.0269230769,'icir':0.0887406876,'dates':260},'2029':{'ic':0.0168654875,'icir':0.0604373552,'dates':68},'latest_120_mature_dates':{'ic':0.0470862471,'icir':0.1615748799,'dates':120},'interpretation':'The 20-day full-sample result passes both binding gates and the low-overlap library screen. Recent annual 2029 ICIR is below the gate, so retain only as a modest conditional sleeve and monitor; latest-120 evidence remains positive. Coverage is event-conditioned at 11.11%.'}
d['last_validated']='2029-05-31'; d['validation_data_cutoff']='2029-05-30'
d['version']='2029-05-31'
json.dump(d,open(p,'w'),indent=2); open(p,'a').write('\n')
