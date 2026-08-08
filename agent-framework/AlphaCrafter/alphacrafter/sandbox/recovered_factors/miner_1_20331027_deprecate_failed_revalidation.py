"""Deprecate failed scheduled revalidation while preserving full prior evidence."""
import json, os
fn='factors/miner_1_20330303_signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d.json'
j=json.load(open(fn,encoding='utf8'))
j['version']='deprecated_revalidation_2033-10-27'
j['validation']['status']='DEPRECATED'
j['validation']['timestamp']='2033-10-27T09:30:00'
j['validation']['period']='2020-01-01 through 2033-10-26; visibility-safe completed-close cutoff'
j['validation']['metrics'].update({'primary_horizon_days':20,'daily_paper_ic':0.016847,'daily_paper_icir':0.054108,'ic_std':0.31137,'ic_standard_error':0.007292,'ic_hit_ratio':0.53209,'ic_dates':1823,'universe_instruments':15,'mean_valid_instruments_per_ic_date':15.0,'signal_cell_coverage':0.503965,'valid_cells':32415,'mean_rank_turnover':0.218109,'turnover_dates':1842,'revalidation_library_evidence_complete':False})
j['validation']['metrics']['decay']={'1d':{'ic':0.016377,'icir':0.053305,'hit_ratio':0.513029,'ic_dates':1842},'5d':{'ic':0.020015,'icir':0.067675,'hit_ratio':0.532644,'ic_dates':1838},'10d':{'ic':0.009241,'icir':0.030797,'hit_ratio':0.518822,'ic_dates':1833},'20d':{'ic':0.016847,'icir':0.054108,'hit_ratio':0.53209,'ic_dates':1823}}
j['validation']['regime_notes']['2033_10_27_revalidation']='Fails binding ICIR gate at every tested horizon: strongest 5d ICIR=0.067675 and 20d ICIR=0.054108, both below 0.084000. 2027 onward 10d ICIR=0.033212. Deprecated; correlation screen is additionally incomplete because no active-library reusable signal artifacts could be mapped.'
j['last_validated']='2033-10-27T09:30:00';j['validation_timestamp']='2033-10-27T09:30:00'
out=fn.replace('.json','_deprecated.json')
with open(out,'w',encoding='utf8') as f:json.dump(j,f,indent=2,ensure_ascii=False)
os.remove(fn)
print('DEPRECATED_TO',out)
