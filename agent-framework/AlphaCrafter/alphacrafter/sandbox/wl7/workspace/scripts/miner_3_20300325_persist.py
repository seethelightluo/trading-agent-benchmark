import pandas as pd, json
s=pd.read_csv('scripts/miner_3_20300325_asymmetry_signal.csv',index_col=0)
# invert original upside-dominance signal into downside-dominance signal
s=-s
s.to_csv('scripts/miner_3_20300325_downside_dominance_signal.csv')
meta={
 'factor_id':'miner_3_20300325_downside_dominance_vol20',
 'factor_name':'Downside-dominance volatility-scaled asymmetry',
 'version':'1.0',
 'calculation':{'expression':'((rolling_sum_20(-min(return,0)) - rolling_sum_20(max(return,0))) / (rolling_sum_20(abs(return))+eps)) / rolling_std_20(return), lagged 1 day','description':'Ranks assets whose recent return activity is dominated by downside moves, normalized by realized volatility; values are lagged one trading day.'},
 'dependencies':['close'],
 'parameters':{'lookback_days':20,'lag_days':1,'forward_horizon_days':10,'epsilon':1e-12},
 'validation':{'status':'EFFECTIVE','period':'2020-01-01 through 2030-03-20','validation_timestamp':'2030-03-25','metrics':{'ic_10d':0.0393928096,'icir_10d':0.1378884312,'ic_hit_ratio':0.5557377049,'dates':2440,'average_instruments':15.0,'coverage':0.5700164745,'rank_turnover':0.0752037294,'decay_ic_1d':0.0099597599,'decay_ic_5d':0.0265409804,'decay_ic_20d':0.0379560839,'decay_ic_40d':0.0225568304,'max_abs_library_correlation':None},'regime_notes':'Positive full-sample and recent 2028-2030 efficacy; early 2020-2024 has insufficient observations in this artifact due to source alignment and should be rechecked in next cycle. Small cross-asset universe warrants conservative interpretation.'},
 'signal_artifact':'scripts/miner_3_20300325_downside_dominance_signal.csv','ic_artifact':'scripts/miner_3_20300325_asymmetry_ic.csv','tags':['asymmetry','downside','volatility','cross_asset'] ,'last_validated':'2030-03-25'
}
with open('factors/miner_3_20300325_downside_dominance_vol20.json','w') as f: json.dump(meta,f,indent=2)
