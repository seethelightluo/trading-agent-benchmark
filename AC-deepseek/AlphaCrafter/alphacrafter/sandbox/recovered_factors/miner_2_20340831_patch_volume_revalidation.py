import json
p='factors/miner_2_20340706_volume_confirmed_intermediate_continuation_20v5x60obs.json'
j=json.load(open(p))
j['version']='2034-08-31 revalidation'
j['last_validated']='2034-08-31'
j['validation']['period']='2020-01-01 to 2034-08-30 (available source panel; realized IC observations remain concentrated through 2028 due source coverage)'
m=j['validation']['metrics'];m['coverage']=0.37508133688257916;m['mean_active_names']=5.626220053238686
j['validation']['regime_notes']='Revalidated 2034-08-31. One-day IC: 2020-22 0.01254 (ICIR 0.02593, n=56); 2023-25 0.17038 (0.46920, n=50); 2026-28 0.02829 (0.07423, n=59). No valid >=8-name IC observations were available for 2029-31 or 2032-34. The source panel has not added usable realized forward-return observations since the prior validation; aggregate one-day evidence is unchanged. It still clears the shared 1d IC/ICIR gates, but evidence is aging and 20d decay remains negative.'
j['benchmark_admission']['selected_metrics']['ic']=0.06600288600288602
j['benchmark_admission']['selected_metrics']['icir']=0.1582194866898446
j['benchmark_admission']['selected_metrics']['metric_path']='validation.metrics.decay.1d'
j['benchmark_admission']['selected_metrics']['quality']=0.010442666470691774
open(p,'w').write(json.dumps(j,indent=2)+'\n')
