import json
p='factors/miner_1_20310403_inverse_residual_downside_range_expansion_exhaustion_60obs.json'
with open(p) as f: d=json.load(f)
old=d['validation']
d['validation_history'].append(old)
m={
 'selected_horizon_days':20,'daily_paper_ic':0.031847,'daily_paper_icir':0.089701,'hit_ratio':0.556630,
 'ic_dates':1448,'mean_valid_instruments':9.555,'minimum_valid_instruments':8,
 'signal_cell_coverage':0.597890,'signal_cells':'14170/23700','daily_rank_turnover':0.046323,
 'concentration_median_cross_sectional_iqr':0.061723,
 'decay':{'1d':{'ic':0.002386,'icir':0.006723,'dates':1467},'5d':{'ic':0.000339,'icir':0.000937,'dates':1463},'10d':{'ic':0.006757,'icir':0.018538,'dates':1458},'20d':{'ic':0.031847,'icir':0.089701,'dates':1448}},
 'max_abs_library_correlation':0.348663,'closest_library_factor':'idiosyncratic_upside_tail_skewness_60obs','library_correlation_evidence_cells':8669,'library_factors_screened':30,
 'correlation_evidence_note':'Unchanged admitted-factor revalidation; the last complete library novelty audit remains valid and is below 0.5000. No new admission is sought.'}
d['version']='2032-08-05'
d['validation']={'period':'2026-07-16 through 2032-08-04 (completed-bar cutoff)','status':'EFFECTIVE','metrics':m,'regime_notes':{
 '2026_2029':{'dates':831,'ic':0.068133,'icir':0.222829,'hit_ratio':0.607702},
 '2030_2032_08_04':{'dates':617,'ic':-0.017023,'icir':-0.041788,'hit_ratio':0.487844},
 'recent_12m':{'dates':202,'ic':-0.104833,'icir':-0.268531,'hit_ratio':0.376238},
 'assessment':'Aggregate 20-day evidence remains above the binding gates (IC 0.031847; ICIR 0.089701), so this incumbent remains EFFECTIVE. Drift is now decisive in the newer samples: post-2030 and recent-12-month IC/ICIR are both negative, and recent hit ratio is only 37.62%. Retain only under enhanced monitoring; do not use recent evidence as supportive. Revalidate early and deprecate immediately if aggregate gates fail.'}}
d['last_validated']='2032-08-05'
d['next_revalidation_due']='2032-09-05'
with open(p,'w') as f: json.dump(d,f,indent=2);f.write('\n')
print('updated',p,'status',d['validation']['status'],'history',len(d['validation_history']))
