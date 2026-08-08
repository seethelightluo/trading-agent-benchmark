import json
p='factors/miner_3_20301226_inverse_vix_persistent_trend_exposure_10v40v50obs.json'
x=json.load(open(p))
m=x['validation']['metrics']
m.update({'primary_horizon_days':10,'daily_paper_ic':0.0628262758857293,'daily_paper_icir':0.2196726395322001,'ic_hit_ratio':0.5962025316455696,'ic_standard_error':0.010175407320139634,'ic_dates':790,'universe_instruments':15,'mean_valid_instruments_per_ic_date':14.99240506329114,'signal_cell_coverage':0.27412497709364114,'mean_instruments_per_panel_date':4.111874656404618,'rank_stability_1d':0.9122292233398585,'implied_rank_turnover_1d':0.08777077666014155,'max_abs_library_correlation':0.30877343335043184,'most_correlated_library_factor':'miner_1_volnorm_reversal_5obs','max_abs_library_correlation_common_signal_cells':1824,'library_correlation_evidence_complete':True,'library_factors_compared':28,'decay_ic':{'1d':0.011215821645441007,'5d':0.03168746092372961,'10d':0.0628262758857293,'20d':0.03627611192814867},'decay_icir':{'1d':0.03846564711907277,'5d':0.10965502133998821,'10d':0.2196726395322001,'20d':0.12488263891615746},'decay_ic_dates':{'1d':790,'5d':790,'10d':790,'20d':790}})
x['validation'].update({'period':'2020-01-01 through 2031-04-30','timestamp':'2031-05-01','status':'EFFECTIVE','regime_notes':'Primary 10-observation horizon: 2024-2026 IC 0.06583, ICIR 0.24228, 46 dates; 2027-2030 IC 0.06017, ICIR 0.20853, 713 dates; 2031 YTD IC 0.11951, ICIR 0.48329, 31 dates. Positive orientation was consistent across all available synchronized regimes; 2031 sample remains short.'})
x['last_validated']='2031-05-01'
x['benchmark_admission']['selected_metrics'].update({'ic':0.0628262758857293,'icir':0.2196726395322001,'max_abs_library_correlation':0.30877343335043184,'quality':0.013801423833723755})
json.dump(x,open(p,'w'),indent=2)
print('updated',p,x['validation']['status'],x['last_validated'])
