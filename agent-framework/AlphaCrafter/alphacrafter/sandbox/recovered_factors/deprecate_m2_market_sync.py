import json
src='factors/miner_2_20261217_market_synchronization_increase_60_20.json'
dst='factors/miner_2_20261217_market_synchronization_increase_60_20_deprecated.json'
with open(src) as fh: d=json.load(fh)
d['version']='2027-09-09'; d['last_validated']='2027-09-09'
d['validation']['period']='2020-01-01 through 2027-09-08; usable IC observations are 2025-2027'
d['validation']['status']='DEPRECATED'
d['validation']['metrics']={'primary_horizon':'20d','daily_paper_ic':-0.042182,'daily_paper_icir':-0.142765,'ic_std':0.295468,'ic_standard_error':0.018838,'ic_hit_ratio':0.463415,'ic_dates':246,'mean_valid_instruments':13.170732,'coverage':0.204092,'mean_rank_turnover':0.104002,'turnover_dates':265,'max_abs_library_correlation':0.199267,'most_correlated_admitted_signal':'miner_2_drawdown_synchronization_improvement_60_20','decay':{'1d':{'ic':0.011844,'icir':0.037532,'hit_ratio':0.528302,'ic_dates':265},'5d':{'ic':0.017116,'icir':0.055049,'hit_ratio':0.547893,'ic_dates':261},'10d':{'ic':0.013014,'icir':0.039321,'hit_ratio':0.542969,'ic_dates':256},'20d':{'ic':-0.042182,'icir':-0.142765,'hit_ratio':0.463415,'ic_dates':246}}}
d['validation']['regime_notes']='All 265 one-day IC dates fall in the available 2025-2027 eligibility regime. The formerly strong short sample did not persist: no horizon now meets the ICIR admission threshold and the selected 20-day orientation is negative. Diversification correlation is acceptable but cannot override failed predictive gates.'
d['validation']['library_correlation_evidence']={'max_abs_library_correlation':0.199267,'most_correlated':'miner_2_drawdown_synchronization_improvement_60_20','comparison_library_signals':17}
d['deprecation_reason']='2027-09-09 re-validation failed binding admission performance gates: 20d |ICIR|=0.142765 with adverse negative orientation; 1d/5d/10d ICIRs were 0.037532/0.055049/0.039321.'
d['benchmark_admission']['deprecated_at']='2027-09-09'
with open(dst,'w') as fh: json.dump(d,fh,indent=2)
print(dst)
