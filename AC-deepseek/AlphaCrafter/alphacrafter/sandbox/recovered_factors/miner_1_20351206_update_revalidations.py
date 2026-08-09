import json
updates={
 'factors/miner_1_20350816_library_orthogonal_severity_weighted_systemic_weakness_downside_persistence_10_60.json':{
  'date':'2035-12-06','cutoff':'2035-12-05','status':'EFFECTIVE_WITH_RECENT_DIRECTION_DRIFT','h':10,'ic':-0.026473,'icir':-0.098770,'recent_ic':0.039365,'recent_icir':0.141944,'corr':0.287843,'coverage':0.500138,'cells':36295,'possible':72570,'turnover':0.135409,'dates':2425},
 'factors/miner_1_20351025_library_orthogonal_drawdown_normalized_severity_systemic_weakness_downside_persistence_10_60.json':{
  'date':'2035-12-06','cutoff':'2035-12-05','status':'EFFECTIVE','h':20,'ic':-0.031399,'icir':-0.101995,'recent_ic':-0.056379,'recent_icir':-0.172987,'corr':0.205345,'coverage':0.367190,'cells':26647,'possible':72570,'turnover':0.121819,'dates':2415}}
for f,u in updates.items():
 with open(f) as q:d=json.load(q)
 m=d['validation']['metrics'];m.update({'effective_horizon_sessions':u['h'],'ic':u['ic'],'icir':u['icir'],'ic_dates':u['dates'],'coverage':u['coverage'],'valid_factor_date_asset_cells':u['cells'],'possible_factor_date_asset_cells':u['possible'],'turnover_mean_absolute_daily_percentile_rank_change':u['turnover'],'max_abs_library_correlation':u['corr'],'audited_library_signal_count':37})
 d['validation']['period']='2020-01-01 to 2035-12-05 (visible completed-session cutoff)';d['last_validated']=u['date']
 d['validation']['status']='EFFECTIVE'
 d.setdefault('revalidation_history',[]).append({'date':u['date'],'visible_data_cutoff':u['cutoff'],'status':u['status'],'selected_horizon_sessions':u['h'],'ic':u['ic'],'icir':u['icir'],'recent_180d_ic':u['recent_ic'],'recent_180d_icir':u['recent_icir'],'max_abs_library_correlation':u['corr']})
 with open(f,'w') as q:json.dump(d,q,indent=2);q.write('\n')
