import json
p='factors/miner_2_20311030_continuous_vix_scaled_inflation_impulse_residual_loading_contraction_60_20d.json'
with open(p,encoding='utf8') as f: d=json.load(f)
m=d['validation']['metrics']
m.update({'daily_paper_ic':0.075887,'daily_paper_icir':0.226747,'ic_std':0.334679,'ic_standard_error':0.008598,'ic_hit_ratio':0.594059,'ic_dates':1515,'mean_valid_instruments':14.993399,'universe_instruments':15,'coverage':0.464957,'valid_factor_cells':27744,'mean_daily_rank_turnover':0.120198,'turnover_dates':1534,'driver_nonzero_fraction':0.378331,'max_abs_library_correlation':0.424268,'most_correlated_library_factor':'residual_positive_oil_change_shock_loading_contraction_20_60d','library_correlation_cells':23100,'library_factors_screened':34,'decay':{'1d':{'ic':0.00574,'icir':0.017778,'ic_dates':1534},'5d':{'ic':0.036383,'icir':0.110883,'ic_dates':1530},'10d':{'ic':0.055581,'icir':0.167962,'ic_dates':1525},'20d':{'ic':0.075887,'icir':0.226747,'ic_dates':1515}}})
d['validation'].update({'period':'2020-01-01 through 2032-08-18; completed-close visibility-safe cutoff','status':'EFFECTIVE','selected_horizon_days':20,'timestamp':'2032-08-19T09:30:00','regime_notes':'At 10 days: 2025-2026 has 66 IC dates, IC +0.032863, ICIR +0.106696, hit 0.515152; 2027 onward has 1,459 dates, IC +0.056609, ICIR +0.170521, hit 0.573681. No 2020-2024 observations survived rolling-history requirements. Full sample clears gates at 5d, 10d and 20d; all IC dates require >=8 assets and average 14.99. Revalidation through 2032-08-18 confirms 20d IC/ICIR and library orthogonality below 0.5.'})
d['last_validated']='2032-08-19'; d['validation_timestamp']='2032-08-19'
d['benchmark_admission']['selected_metrics'].update({'ic':0.075887,'icir':0.226747,'max_abs_library_correlation':0.424268,'quality':0.017208})
with open(p,'w',encoding='utf8') as f: json.dump(d,f,indent=2);f.write('\n')
print('updated',p,'status',d['validation']['status'],'validated',d['last_validated'])
