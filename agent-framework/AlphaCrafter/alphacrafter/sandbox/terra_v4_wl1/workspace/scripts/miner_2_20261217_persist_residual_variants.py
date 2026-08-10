import numpy as np,pandas as pd, json
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
 P[s]=d[d.index<=cut]
P=pd.DataFrame(P).sort_index(); R=P.pct_change(); med=R.median(axis=1); rv=R.rolling(20,min_periods=15).std()*np.sqrt(20)
for lag,ic,icir,decay in [(4,0.04224111384820724,0.12703781163987407,0.028244677850755504),(6,0.03431058985434704,0.1037468353698027,0.018137967354063186)]:
 F=-(P.pct_change(lag).sub(P.pct_change(lag).median(axis=1),axis=0)).div(rv.replace(0,np.nan))
 out=F.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; path=f'scripts/miner_2_20261217_residual_reversal_{lag}d_signal.csv'; out.to_csv(path,index=False)
 obj={'factor_id':f'miner_2_20261217_residual_reversal_{lag}d','factor_name':f'Volatility-normalized {lag}-day cross-sectional residual reversal','version':'2026-12-17.1','calculation':{'expression':f'-(return_{lag}d - cross_sectional_median(return_{lag}d)) / (rolling_std_daily_return_20d * sqrt(20))','description':'Ranks assets higher when their recent return trails the cross-sectional median, normalized by trailing realized volatility; all inputs are lagged completed-day prices.'},'dependencies':['close','daily_return'],'parameters':{'lookback_days':lag,'volatility_window':20,'min_vol_observations':15},'validation':{'status':'EFFECTIVE','period':'2020-01-01 to 2026-12-17','metrics':{'daily_ic':ic,'daily_icir':icir,'five_day_ic':decay,'coverage':0.9918769630672587,'turnover':None,'max_abs_library_correlation':None,'dates':1781,'average_instruments':14.5682201010668},'regime_notes':'Positive daily predictive IC across the full synthetic history; small 15-asset cross-section warrants conservative uncertainty interpretation.','signal_artifact':path},'tags':['cross_asset','reversal','volatility_normalized'],'last_validated':'2026-12-17T00:00:00Z'}
 jpath=f'factors/{obj["factor_id"]}.json'
 with open(jpath,'w') as f: json.dump(obj,f,indent=2)
 with open(jpath) as f: chk=json.load(f)
 print(jpath,chk['factor_id'],chk['validation']['status'],chk['validation']['metrics']['daily_ic'],chk['validation']['metrics']['daily_icir'],chk['validation']['signal_artifact'])
