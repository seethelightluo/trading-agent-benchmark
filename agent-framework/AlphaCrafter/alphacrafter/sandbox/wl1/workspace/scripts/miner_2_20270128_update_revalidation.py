import json
p='factors/miner_2_20261231_corr_adjusted_trend_20d.json'
with open(p,encoding='utf-8') as f:d=json.load(f)
v=d['validation']; v['last_validated']='2027-01-28T00:00:00Z'; v['period']='2020-01-01 through 2027-01-27'; v['regime_notes']='Revalidated through 2027-01-27. 10d IC/ICIR 0.0439254/0.1403185; 20d IC/ICIR 0.0512234/0.1586615. Annual IC remained positive 2020-2025 but was negative in 2026 (-0.0563); the nine available 2027 10d observations were negative (-0.1143), so retain with conservative monitoring.'
m=v['metrics'];m.update({'primary_horizon_days':20,'ic':0.0512233929,'icir':0.1586614525,'hit_rate':0.5573505654,'valid_dates':2476,'average_instruments':14.9979806,'coverage':0.9872627,'rank_turnover':0.0582185,'max_abs_library_correlation':None,'decay':{'10d_ic':0.0439254033,'10d_icir':0.1403185332,'20d_ic':0.0512233929,'20d_icir':0.1586614525}})
with open(p,'w',encoding='utf-8') as f:json.dump(d,f,indent=2)
with open(p,encoding='utf-8') as f:x=json.load(f)
print(x['factor_id'],x['validation']['status'],x['validation']['metrics']['ic'],x['validation']['metrics']['icir'],x['validation']['last_validated'],x['calculation'].get('signal_provenance'))
