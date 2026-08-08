import json,os
src='factors/miner_3_20261217_high_vix_momentum_residual_downside_asymmetry_20.json'
dst='factors/miner_3_20261217_high_vix_momentum_residual_downside_asymmetry_20_deprecated.json'
with open(src,encoding='utf-8') as f: d=json.load(f)
d['validation']={'period':'2020-01-01 through 2027-06-30; data visible only through 2027-06-30','status':'DEPRECATED','metrics':{'ic':0.002855,'icir':0.009099,'ic_horizon':'10d','paper_ic':{'1d':0.001682,'5d':0.012395,'10d':0.002855,'20d':-0.021161},'paper_icir':{'1d':0.005437,'5d':0.037887,'10d':0.009099,'20d':-0.066624},'hit_ratio':{'1d':0.5156,'5d':0.4922,'10d':0.5099,'20d':0.4654},'ic_dates':{'1d':609,'5d':386,'10d':353,'20d':492},'mean_valid_instruments':{'1d':14.25,'5d':14.14,'10d':14.17,'20d':14.08},'ic_standard_error':{'1d':0.012536,'5d':0.016652,'10d':0.016701,'20d':0.014320},'signal_cell_coverage':0.510943,'mean_daily_rank_turnover':None,'max_abs_library_correlation':0.414089,'max_abs_library_correlation_factor':'miner_2_20270520_inverse_residual_return_skewness_20obs','library_correlation_common_cells':2880},'regime_notes':'Revalidation fails the binding predictive gates at every tested horizon. Selected 10d IC is 0.002855 and ICIR 0.009099; 5d ICIR is 0.037887. Ten-day results are weak/unstable in each available segment: 2020-21 0.014921/0.046784 (70 dates), 2022-23 -0.017647/-0.059598 (71), 2024-25 0.010986/0.040787 (66), 2026-27 0.003364/0.009901 (146). The correlation test remains diversified (0.414089 < 0.5), but cannot compensate for failed IC and ICIR gates.'}
d['last_validated']='2027-07-01';d['deprecation_reason']='Failed periodic revalidation: no same-horizon IC/ICIR pair meets benchmark admission gates.'
d['benchmark_admission']['revalidated_at']='2027-07-01';d['benchmark_admission']['revalidation_passed']=False
with open(dst,'w',encoding='utf-8') as f: json.dump(d,f,indent=2)
os.remove(src)
print('deprecated_record',dst)
