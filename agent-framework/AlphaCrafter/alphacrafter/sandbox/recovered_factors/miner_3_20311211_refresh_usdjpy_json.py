import json
p='factors/miner_3_20300711_conditional_usdjpy_impulse_exposure_10v50obs.json'
j=json.load(open(p)); m=j['validation']['metrics']
decay={
'1':{'daily_paper_ic':0.003024061167959276,'daily_paper_icir':0.010193296317615256,'ic_hit_ratio':0.5136494252873564,'ic_standard_error':0.00795163258101935,'ic_dates':1392,'mean_valid_instruments':14.995689655172415},
'5':{'daily_paper_ic':0.01868637700357802,'daily_paper_icir':0.06260317303992953,'ic_hit_ratio':0.5201729106628242,'ic_standard_error':0.008011872467409932,'ic_dates':1388,'mean_valid_instruments':14.995677233429396},
'10':{'daily_paper_ic':0.024372171574010122,'daily_paper_icir':0.08529680634015374,'ic_hit_ratio':0.5292841648590022,'ic_standard_error':0.007683345558038013,'ic_dates':1383,'mean_valid_instruments':14.995661605206074},
'20':{'daily_paper_ic':0.014651762590824532,'daily_paper_icir':0.04874166897046689,'ic_hit_ratio':0.5054624908958485,'ic_standard_error':0.008112490636024997,'ic_dates':1373,'mean_valid_instruments':14.99563000728332}}
m.update({'primary_horizon_observations':10,**decay['10'],'panel_dates':3798,'signal_coverage':0.4502720730208881,'mean_active_names_per_date':6.754081095313323,'rank_stability_1d':0.6849577982525125,'implied_rank_turnover_proxy':0.31504220174748754,'decay':decay,'max_abs_library_correlation':0.2590346024416182,'most_correlated_library_factor':'miner_3_dxy_vix_state_impulse_exposure_5v40v20obs','library_correlation_common_signal_cells':None,'library_correlation_evidence_complete':True})
j['version']='2031-12-11'; j['last_validated']='2031-12-11'; j['validation'].update({'period':'2020-01-01 through 2031-12-10','timestamp':'2031-12-11','status':'EFFECTIVE','regime_notes':'10-observation IC: 2024–2026 -0.10760 (ICIR -0.53021; 104 dates), 2027–2030 0.04444 (0.15259; 1,043 dates), 2031 YTD -0.00616 (-0.02239; 236 dates). Full-history IC/ICIR narrowly clears admission gates, but 2031 has turned negative; retain EFFECTIVE under close monitoring and reduce deployment priority rather than expand exposure.'})
s=j['benchmark_admission']['selected_metrics'];s.update({'ic':decay['10']['daily_paper_ic'],'icir':decay['10']['daily_paper_icir'],'max_abs_library_correlation':0.2590346024416182,'quality':decay['10']['daily_paper_ic']*decay['10']['daily_paper_icir']})
open(p,'w').write(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
print('updated',p,'quality',s['quality'])
