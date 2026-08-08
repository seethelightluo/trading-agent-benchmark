import json, os
p='factors/miner_3_20270311_volatility_compression_5v60.json'
with open(p,encoding='utf-8') as f: d=json.load(f)
d['version']='2028-01-27'
d['last_validated']='2028-01-27'
d['validation']['period']='visible data through 2028-01-26; aligned usable IC panel 2026-2028'
d['validation']['timestamp']='2028-01-27'
d['validation']['status']='DEPRECATED'
d['validation']['metrics'].update({
 'same_horizon':'1d','ic':0.002543,'icir':0.009651,'hit_ratio':0.505376,'ic_dates':372,
 'mean_valid_instruments':14.98,'ic_standard_error':0.013660,'coverage':0.247585,
 'mean_daily_rank_turnover':0.162321,'concentration':'Broad cross-sectional ranking: mean 14.98 of 15 instruments per IC date.',
 'decay':{'1d_ic':0.002543,'1d_icir':0.009651,'5d_ic':-0.010017,'5d_icir':-0.041249,'10d_ic':-0.010230,'10d_icir':-0.039603,'20d_ic':-0.016070,'20d_icir':-0.066305}
})
d['validation']['regime_notes']='Scheduled revalidation used 372 usable daily 1d IC dates with 14.98 instruments on average, spanning the available aligned 2026-2028 panel (no earlier aligned observations available). The formerly admitted positive 1d effect has disappeared: IC 0.002543 and ICIR 0.009651. No tested horizon meets the binding absolute IC and ICIR gates; 20d is negative but also fails stability (ICIR -0.066305). Deprecated on 2028-01-27; do not use in active ensemble unless a future independent revalidation readmits it.'
d['benchmark_admission']['revalidation']={'timestamp':'2028-01-27','passed':False,'failure_reason':'No horizon met |IC| >= 0.0070 and |ICIR| >= 0.0840; selected 1d IC=0.002543, ICIR=0.009651.'}
out='factors/miner_3_20270311_volatility_compression_5v60_deprecated.json'
with open(out,'w',encoding='utf-8') as f: json.dump(d,f,indent=2); f.write('\n')
os.remove(p)
print('deprecated record written:',out)
