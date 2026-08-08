import json
p='factors/miner_1_20310403_inverse_residual_downside_range_expansion_exhaustion_60obs.json'
with open(p) as h:d=json.load(h)
m=d['validation']['metrics']
m.update({'selected_horizon_days':20,'daily_paper_ic':0.067475,'daily_paper_icir':0.205478,'hit_ratio':0.5926,'ic_dates':1188,'mean_valid_instruments':9.75,'signal_cell_coverage':0.301955,'signal_cells':'16523/54720','daily_rank_turnover':0.043181,'concentration_mean_cross_sectional_sd':0.067435,'decay':{'1d':{'ic':0.014216,'icir':0.041194,'dates':1207},'5d':{'ic':0.02078,'icir':0.05898,'dates':1203},'10d':{'ic':0.027794,'icir':0.078855,'dates':1198},'20d':{'ic':0.067475,'icir':0.205478,'dates':1188}}})
d['validation']['period']='2020-01-01 through 2031-05-14 (visible-through date)'
d['validation']['status']='EFFECTIVE'
d['validation']['regime_notes']['2026_current']={'dates':1188,'ic':0.067475,'icir':0.205478,'hit_ratio':0.5926}
d['validation']['regime_notes']['assessment']='Revalidation retains admission at the 20-day horizon: both shared IC gates remain satisfied, though IC and ICIR have declined versus 2031-04-03. All usable observations remain post-2026. The original complete 30-signal admission novelty audit (maximum correlation 0.348663) remains the recorded library evidence; signal-level novelty was not recomputed in this revalidation because admitted JSON definitions do not provide generic signal panels. Retain conservatively and revalidate by 2031-08-15.'
d['last_validated']='2031-05-15'
with open(p,'w') as h:json.dump(d,h,indent=2);h.write('\n')
print('updated',p)
