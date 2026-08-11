import json
p='factors/miner_2_20261231_corr_adjusted_trend_20d.json'
with open(p,encoding='utf-8') as f:d=json.load(f)
d['calculation']['signal_provenance']='Computed from close prices through 2027-01-27 only; equal-weight equity basket is 000300.SH, SPX, HSI, N225, SX5E, 000688.SH, SOX, NDX. No observation-only macro series or future data used.'
with open(p,'w',encoding='utf-8') as f:json.dump(d,f,indent=2)
with open(p,encoding='utf-8') as f:x=json.load(f)
assert x['factor_id']=='miner_2_20261231_corr_adjusted_trend_20d' and x['validation']['status']=='EFFECTIVE' and '2027-01-27' in x['calculation']['signal_provenance']
print('verified',x['factor_id'],x['validation']['status'],x['calculation']['signal_provenance'])
