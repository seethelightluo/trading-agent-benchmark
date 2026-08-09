"""Deprecate residual upside serial reversal after scheduled 2028-12-28 validation."""
import json,pathlib
p=pathlib.Path('factors/miner_2_20280601_residual_upside_serial_reversal_60d.json')
d=json.loads(p.read_text())
d['version']='2028-12-28'
d['last_validated']='2028-12-28T09:30:00'
v=d['validation'];v['timestamp']='2028-12-28T09:30:00';v['status']='DEPRECATED'
v['period']='2020-01-01 through 2028-12-27; eligible IC observations begin in 2025 because of aligned-calendar coverage and rolling requirements'
v['metrics']={'primary_horizon_days':20,'daily_paper_ic':-0.039752,'daily_paper_icir':-0.125668,'ic_std':0.316328,'ic_standard_error':0.013355,'ic_hit_ratio':0.454545,'ic_dates':561,'universe_instruments':15,'mean_valid_instruments_per_ic_date':10.500891,'signal_cell_coverage':0.238221,'mean_rank_turnover':0.038880,'turnover_dates':580,'concentration_note':'Continuous residual-autocorrelation signal; 561 IC dates satisfy the at-least-8-name rule, averaging 10.50 valid instruments; latest signal has 10 valid instruments.','max_abs_library_correlation':0.193662,'max_abs_library_correlation_factor':'miner_3_residual_lower_partial_moment_60d','library_signal_cells_compared':10820,'decay':{'1d':{'ic':-0.006953,'icir':-0.021098,'hit_ratio':0.493103,'dates':580,'mean_valid_instruments':10.4845},'5d':{'ic':-0.015586,'icir':-0.046513,'hit_ratio':0.479167,'dates':576,'mean_valid_instruments':10.4878},'10d':{'ic':-0.020970,'icir':-0.064748,'hit_ratio':0.465849,'dates':571,'mean_valid_instruments':10.4921},'20d':{'ic':-0.039752,'icir':-0.125668,'hit_ratio':0.454545,'dates':561,'mean_valid_instruments':10.5009}}}
v['regime_notes']={'2025_2026':{'ic_20d':-0.113459,'icir_20d':-0.435818,'hit_ratio_20d':0.370968,'ic_dates':62},'2027_2028_12_27':{'ic_20d':-0.030595,'icir_20d':-0.095117,'hit_ratio_20d':0.464930,'ic_dates':499},'interpretation':'The 20-session negative orientation still clears absolute IC and ICIR numerically, but scheduled revalidation shows substantial magnitude deterioration versus the prior validation and all same-horizon short decay metrics fail their gates. Conservatively deprecated for performance drift; not available to the active ensemble pending a future re-admission.'}
out=p.with_name(p.stem+'_deprecated.json')
out.write_text(json.dumps(d,indent=2)+'\n')
p.unlink()
print('DEPRECATED',out)
