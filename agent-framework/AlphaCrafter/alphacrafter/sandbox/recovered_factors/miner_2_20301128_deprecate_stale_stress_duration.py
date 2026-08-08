import json,os
p='factors/miner_2_20290614_stress_duration_weighted_peer_resilience_reversal_60.json'
d=json.load(open(p))
m=d['validation']['metrics']
m.update({'universe_size':15,'factor_cells':34864,'total_cells':52920,'coverage':0.658806,'mean_valid_instruments_per_date':14.292,'minimum_ic_cross_section':8,'stress_session_share':0.199546,'turnover_mean_daily_rank_change':0.01586,'mean_cross_sectional_sd':0.010151,'validation_endpoint':'2030-03-21','data_cursor_endpoint':'2030-11-27','factor_last_valid_date':'2030-03-21','revalidation_failure_reason':'No valid signal cross-sections after 2030-03-21 under the required minimum stress-event condition; thus current effectiveness cannot be established despite historical 20d gate passing.'})
d['validation']['status']='DEPRECATED'
d['validation']['period']='2020-01-01 to 2030-03-21 (factor-valid observations; data cursor 2030-11-27)'
d['validation']['regime_notes']='Revalidation run 2030-11-28 with 15 tradable assets and data visible through 2030-11-27. Historical 20d IC/ICIR remains +0.048601/+0.137694 across 1,659 daily IC dates (mean 14.292 instruments), but the conditional signal has no valid broad cross-section after 2030-03-21 because the rolling minimum stress-event condition is unmet. The signal is stale by over eight months and fails timeliness/current-coverage revalidation; it is deprecated rather than retained as active. Historical library novelty evidence (+0.481433 versus downside_cross_asset_beta_resilience_40) is retained only as provenance, not refreshed.'
d['last_validated']='2030-11-28'
d['deprecation']={'deprecated_at':'2030-11-28','reason':'timeliness_and_current_signal_coverage_failure','historical_20d_ic':0.048601,'historical_20d_icir':0.137694,'factor_last_valid_date':'2030-03-21','data_cursor_endpoint':'2030-11-27'}
out=p.replace('.json','_deprecated.json')
with open(out,'w') as f: json.dump(d,f,indent=2)
os.remove(p)
print(out)
