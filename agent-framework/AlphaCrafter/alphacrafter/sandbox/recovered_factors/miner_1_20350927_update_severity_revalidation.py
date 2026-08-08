import json
p='factors/miner_1_20350816_library_orthogonal_severity_weighted_systemic_weakness_downside_persistence_10_60.json'
with open(p,encoding='utf-8') as f: x=json.load(f)
m=x['validation']['metrics']
m.update({'effective_horizon_sessions':10,'ic':-0.028527,'icir':-0.106637,'ic_hit_ratio':0.446737,'ic_dates':2375,'mean_instruments_per_ic_date':14.903,'minimum_instruments_per_ic_date':10,'coverage':0.494918,'valid_factor_date_asset_cells':35545,'possible_factor_date_asset_cells':71820,'turnover_mean_absolute_daily_percentile_rank_change':0.135720,'concentration_mean_cross_sectional_std':0.107313,'max_abs_library_correlation':0.290230,'most_correlated_library_factor':'peer_relative_downside_volatility_compression_10_40','library_correlation_aligned_observations':35545,'audited_library_signal_count':37,'decay':{'1_sessions':{'ic':-0.010207,'icir':-0.037387,'hit_ratio':0.483221,'dates':2384},'5_sessions':{'ic':-0.015282,'icir':-0.055883,'hit_ratio':0.470588,'dates':2380},'10_sessions':{'ic':-0.028527,'icir':-0.106637,'hit_ratio':0.446737,'dates':2375},'20_sessions':{'ic':-0.021939,'icir':-0.080643,'hit_ratio':0.472727,'dates':2365}}})
x['validation']['period']='2020-01-01 to 2035-09-26 (visible completed-session cutoff)'
x['validation']['status']='EFFECTIVE'
x['validation']['regime_notes']={'2023_2026':{'ic':-0.101950,'icir':-0.416202,'hit_ratio':0.320755,'dates':106},'2027_2035_09_26':{'ic':-0.025097,'icir':-0.093617,'hit_ratio':0.452622,'dates':2269},'recent_180_calendar_days':{'ic':-0.015893,'icir':-0.058755,'hit_ratio':0.495798,'dates':119},'note':'Full-history and post-2026 inverse-direction 10-session evidence remain admission-compliant, and library independence remains strong. Recent 180-day ICIR has weakened below the absolute stability gate; retain as effective but reduce confidence and revalidate next quarterly cycle.'}
x['last_validated']='2035-09-27'
x['admission']['quality_score_abs_ic_times_abs_icir']=round(abs(m['ic'])*abs(m['icir']),10)
x['revalidation_history']=x.get('revalidation_history',[])+[{'date':'2035-09-27','visible_data_cutoff':'2035-09-26','status':'EFFECTIVE_WITH_RECENT_DRIFT','selected_horizon_sessions':10,'ic':-0.028527,'icir':-0.106637,'recent_180d_ic':-0.015893,'recent_180d_icir':-0.058755,'max_abs_library_correlation':0.290230}]
with open(p,'w',encoding='utf-8') as f: json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
print('updated',p,x['last_validated'],m['max_abs_library_correlation'])
