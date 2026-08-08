import json, os
p='factors/miner_1_20310403_inverse_residual_downside_range_expansion_exhaustion_60obs.json'
with open(p) as f:d=json.load(f)
d.setdefault('validation_history',[]).append(d['validation'])
m={'selected_horizon_days':20,'daily_paper_ic':0.028287,'daily_paper_icir':0.079564,'hit_ratio':0.551075,'ic_dates':1488,'mean_valid_instruments':9.513,'minimum_valid_instruments':8,'signal_cell_coverage':0.596296,'signal_cells':'14490/24300','daily_rank_turnover':0.046595,'concentration_median_cross_sectional_iqr':0.060930,'decay':{'1d':{'ic':0.001469,'icir':0.004125,'dates':1507},'5d':{'ic':0.001265,'icir':0.003482,'dates':1503},'10d':{'ic':0.009707,'icir':0.026448,'dates':1498},'20d':{'ic':0.028287,'icir':0.079564,'dates':1488}},'max_abs_library_correlation':0.348663,'closest_library_factor':'idiosyncratic_upside_tail_skewness_60obs','library_correlation_evidence_cells':8669,'library_factors_screened':30,'correlation_evidence_note':'No new admission is sought. Latest complete incumbent novelty audit (2032-09-02) was below 0.5000; factor is deprecated for failing predictive gates, not correlation.'}
d['version']='2032-09-30'
d['validation']={'period':'2026-07-16 through 2032-09-29 (completed-bar cutoff)','status':'DEPRECATED','metrics':m,'regime_notes':{'2026_2029':{'dates':831,'ic':0.068133,'icir':0.222829,'hit_ratio':0.607702},'2030_2032_09_29':{'dates':657,'ic':-0.022112,'icir':-0.054656,'hit_ratio':0.479452},'recent_12m':{'dates':202,'ic':-0.121535,'icir':-0.312073,'hit_ratio':0.341584},'assessment':'Aggregate 20-day IC remains positive and exceeds 0.0070, but ICIR 0.079564 is below the binding 0.0840 gate. Post-2030 and recent-12-month evidence is negative. Deprecated under the binding periodic revalidation rule.'}}
d['last_validated']='2032-09-30';d['deprecated_on']='2032-09-30';d['deprecation_reason']='20-day aggregate ICIR 0.079564 below binding 0.0840; adverse post-2030 and recent-12-month drift.'
out=p.replace('.json','_deprecated.json')
with open(out,'w') as f:json.dump(d,f,indent=2);f.write('\n')
os.remove(p)
print('deprecated',out,'history',len(d['validation_history']))
