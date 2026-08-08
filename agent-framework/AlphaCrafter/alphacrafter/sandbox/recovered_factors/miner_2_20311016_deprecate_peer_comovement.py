import json,os
p='factors/miner_2_20310515_inverse_peer_up_down_comovement_asymmetry_60obs.json'
with open(p) as f:x=json.load(f)
m=x['validation']['metrics']
m.update({'daily_paper_ic':0.030890833990007537,'daily_paper_icir':0.1029896777787029,'ic_hit_ratio':0.5316804407713499,'ic_standard_error':0.006426980710358533,'ic_dates':2178,'mean_valid_instruments':13.707070707070708,'coverage':0.8596061734965407,'mean_available_signals':12.894092602448112,'rank_stability_1d':0.9620525401076816,'implied_rank_turnover':0.0379474598923184,'max_abs_library_correlation':0.23376111910444944,'most_correlated_library_factor':'miner_1_inverted_downside_cross_asset_beta_40d','library_correlation_evidence_complete':True,'revalidation_recent_10d_ic':-0.08944220336625398,'revalidation_recent_10d_icir':-0.38906760301325716,'revalidation_recent_10d_dates':79})
x['validation']['period']='2020-01-01 to 2031-10-15'
x['validation']['status']='DEPRECATED'
x['validation']['decay']={'1d':{'ic':0.007885741805761222,'icir':0.02620869954556329,'dates':2456},'5d':{'ic':0.020273430697742626,'icir':0.06816501682800943,'dates':2180},'10d':{'ic':0.030890833990007537,'icir':0.1029896777787029,'dates':2178},'20d':{'ic':0.01353108219637994,'icir':0.04410697756249293,'dates':2435}}
x['validation']['regime_notes']='Ten-day revalidation: 2020-21 +0.06203/+0.19060 (230 dates); 2022-23 -0.02862/-0.08610 (262); 2024-26 +0.04863/+0.15849 (468); 2027-30 +0.03809/+0.13106 (1,043); 2031 YTD -0.01127/-0.04924 (175). Although full-sample 10d IC/ICIR passes (+0.03089/+0.10299), post-prior-validation 2031-05-15 onward is -0.08944/-0.38907 over 79 dates. Material recent reversal fails timeliness/drift revalidation; deprecated rather than retained as active.'
x['last_validated']='2031-10-16'
x['benchmark_admission']['revalidation_decision']='DEPRECATED: recent 10d ICIR negative and IC materially negative despite passing long-history aggregate.'
new=p.replace('.json','_deprecated.json')
with open(new,'w') as f:json.dump(x,f,indent=2)
os.remove(p)
print(new)
