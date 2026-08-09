import json,os
old='factors/miner_2_20300530_downside_close_location_participation_recovery_residual_20.json'
new=old[:-5]+'_deprecated.json'
d=json.load(open(old,encoding='utf-8'))
d['validation']['status']='DEPRECATED'
d['validation']['period']='2026-07-16 to 2030-08-07 (visible-data evaluation; forward returns available only within research cursor)'
m=d['validation']['metrics']
m.update({'ic':0.037011,'icir':0.124779,'ic_horizon_days':10,'ic_dates':990,'hit_ratio':0.5394,'mean_instruments':11.75,'ic_standard_error':0.009427,'turnover_mean_daily_rank':0.149186,'coverage':0.226875,'valid_factor_cells':11734,'concentration':'Cross-sectional IC uses all 15 benchmark instruments; selected-horizon observations averaged 11.75 valid instruments and every included date had at least 8.','decay':{'1_day':{'ic':0.015612,'icir':0.052249,'dates':999,'hit_ratio':0.5155},'5_day':{'ic':0.0211,'icir':0.071466,'dates':995,'hit_ratio':0.5337},'10_day':{'ic':0.037011,'icir':0.124779,'dates':990,'hit_ratio':0.5394},'20_day':{'ic':0.024197,'icir':0.083712,'dates':980,'hit_ratio':0.5429}},'max_abs_library_correlation':0.997007,'closest_library_factor':'continuous_participation_weighted_rebound_residual_20','closest_common_valid_cells':11734})
d['validation']['regime_notes']='10-day revalidation: 2026-2027: 360 dates, IC 0.062005, ICIR 0.208712, hit 0.5806; 2028-2030-08-07: 630 dates, IC 0.022728, ICIR 0.076882, hit 0.5159. The same-horizon IC and ICIR continue to clear their numerical gates, but full 32-signal library reconstruction found |rho|=0.997007 versus continuous_participation_weighted_rebound_residual_20 on 11,734 common valid cells. This breaches the strict <0.5000 novelty contract; factor deprecated rather than retained as duplicative.'
d['last_validated']='2030-08-08'; d['validation_timestamp']='2030-08-08';d['revalidation_due']=None
d['benchmark_admission']['selected_metrics'].update({'ic':0.037011,'icir':0.124779,'max_abs_library_correlation':0.997007,'quality':0.004618166169})
json.dump(d,open(new,'w',encoding='utf-8'),indent=2)
os.remove(old)
print(new)
