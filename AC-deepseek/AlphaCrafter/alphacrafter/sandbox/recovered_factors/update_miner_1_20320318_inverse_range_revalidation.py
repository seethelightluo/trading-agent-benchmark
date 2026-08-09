import json
p='factors/miner_1_20310403_inverse_residual_downside_range_expansion_exhaustion_60obs.json'
with open(p) as f: d=json.load(f)
old=d['validation']
d.setdefault('validation_history',[]).append(old)
metrics={
 'selected_horizon_days':20,'daily_paper_ic':0.043722,'daily_paper_icir':0.12505,'hit_ratio':0.569164,
 'ic_dates':1388,'mean_valid_instruments':9.622,'minimum_valid_instruments':8,
 'signal_cell_coverage':0.610045,'signal_cells':'13543/22200','daily_rank_turnover':0.046027,
 'concentration_median_cross_sectional_iqr':0.061548,
 'decay':{'1d':{'ic':0.007496,'icir':0.021415,'dates':1407},'5d':{'ic':0.009331,'icir':0.026071,'dates':1403},'10d':{'ic':0.018577,'icir':0.051817,'dates':1398},'20d':{'ic':0.043722,'icir':0.12505,'dates':1388}},
 'max_abs_library_correlation':0.348663,'closest_library_factor':'idiosyncratic_upside_tail_skewness_60obs','library_correlation_evidence_cells':8669,'library_factors_screened':30,
 'correlation_evidence_note':'Unchanged-factor revalidation; complete admission novelty audit retained from prior validation.'}
d['validation']={'period':'2026-07-16 through 2032-03-17 (completed-bar cutoff)','status':'EFFECTIVE','metrics':metrics,'regime_notes':{
 '2026_2029':{'dates':831,'ic':0.068133,'icir':0.222829,'hit_ratio':0.607702},
 '2030_2032_03_17':{'dates':557,'ic':0.007303,'icir':0.01808,'hit_ratio':0.51167},
 'recent_12m':{'dates':242,'ic':-0.068477,'icir':-0.153766,'hit_ratio':0.442149},
 'assessment':'Aggregate 20-day evidence passes both shared gates (IC 0.043722; ICIR 0.125050), and the two broad partitions remain positive. However, the recent 12-month slice has turned negative. Retain EFFECTIVE under the aggregate revalidation rule, but flag material drift and reassess at the next quarterly review (or earlier if portfolio monitoring weakens).'}}
d['version']='2032-03-18';d['last_validated']='2032-03-18';d['next_revalidation_due']='2032-06-18'
with open(p,'w') as f: json.dump(d,f,indent=2); f.write('\n')
print('updated',p,'history entries',len(d['validation_history']))
