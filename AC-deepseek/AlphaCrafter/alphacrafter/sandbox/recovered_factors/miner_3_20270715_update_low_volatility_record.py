import json
p='factors/miner_3_20270225_low_volatility_of_volatility_20obs.json'
d=json.load(open(p))
d['version']='2027-07-15'
d['last_validated']='2027-07-15'
d['validation']['period']='2020-01-01 through 2027-07-14; effective signal/forward-return sample is 2026-08-03 through 2027-07-14, data visible only through 2027-07-14'
d['validation']['status']='EFFECTIVE'
d['validation']['metrics']={'ic':0.0300590339,'icir':0.0895138188,'ic_horizon':'10d','paper_ic':{'1d':0.0040103272,'5d':0.011106828,'10d':0.0300590339,'20d':0.0368589306},'paper_icir':{'1d':0.0121099046,'5d':0.0359685565,'10d':0.0895138188,'20d':0.1207293025},'hit_ratio':{'1d':0.4959349593,'5d':0.4958677686,'10d':0.5274261603,'20d':0.5374449339},'ic_dates':{'1d':246,'5d':242,'10d':237,'20d':227},'mean_valid_instruments':{'1d':12.1341463415,'5d':12.1363636364,'10d':12.1392405063,'20d':12.1453744493},'ic_standard_error':{'1d':0.0211140491,'5d':0.0198499471,'10d':0.021812767,'20d':0.0202636236},'signal_cell_coverage':0.195443,'mean_daily_rank_turnover':0.055198,'max_abs_library_correlation':0.309422,'max_abs_library_correlation_factor':'miner_2_peer_crowding_correlation_20obs','library_correlation_common_cells':7755,'library_size_at_validation':14}
d['validation']['regime_notes']='All valid IC observations remain in the 2026-27 available aligned panel (237 10d dates: IC 0.030059, ICIR 0.089514, hit ratio 52.74%). The selected 10d horizon now narrowly clears both binding gates; 20d is stronger (IC 0.036859, ICIR 0.120729). No aligned earlier-regime observations exist, so evidence remains recent-panel limited and warrants revalidation by 2027-10-15.'
d['benchmark_admission']['selected_metrics'].update({'ic':0.0300590339,'icir':0.0895138188,'max_abs_library_correlation':0.309422,'quality':0.002690,'metric_path':'validation.metrics','correlation_path':'validation.metrics.max_abs_library_correlation'})
json.dump(d,open(p,'w'),indent=2)
print('updated',p)
