import json
p='factors/miner_2_20341012_volume_shock_short_reversal_5v20x60obs.json'
j=json.load(open(p)); j['version']='2035-01-04 revalidation'; j['last_validated']='2035-01-04'
j['validation']['period']='2020-01-01 to 2035-01-03 (visible source panel; valid realized IC evidence occurs only in 2026-2028)'
j['validation']['status']='EFFECTIVE'
m=j['validation']['metrics'];m.update({'daily_paper_ic':0.06522988505747127,'daily_paper_icir':0.1859809731033574,'ic_hit_ratio':0.5517241379310345,'ic_standard_error':0.04605365473588679,'ic_dates':58,'mean_valid_instruments':9.0,'coverage':0.35629984051036684,'mean_active_names':5.344497607655502,'mean_rank_stability_1d':0.5967836257309941,'implied_rank_turnover':0.4032163742690059})
j['validation']['regime_notes']='Revalidated 2035-01-04 using data visible through 2035-01-03. All 58 eligible one-day IC observations remain in 2026-2028 (IC 0.06523; ICIR 0.18598; hit 55.17%). No >=8-name realized IC dates were available in 2020-25, 2029-34, or the available 2035 sessions, so no fresh realized out-of-sample evidence was added. It remains EFFECTIVE under the shared historical gates, but its evidence is temporally concentrated and should receive a cautious subsequent revalidation.'
json.dump(j,open(p,'w'),indent=2);open(p,'a').write('\n')
