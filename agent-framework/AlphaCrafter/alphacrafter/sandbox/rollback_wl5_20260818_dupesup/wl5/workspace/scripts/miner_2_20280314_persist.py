import os,json,numpy as np,pandas as pd
base='../persistent/stock_data'; syms=sorted(x[:-4] for x in os.listdir(base) if x.endswith('.csv'))
px={}
for s in syms:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); vol5=r.rolling(5,min_periods=5).std(); vol20=r.rolling(20,min_periods=20).std()
sig=-(P.pct_change(5))*(vol5/(vol20+1e-12)).clip(0,5)
sig.index.name='date'; sig.to_csv('scripts/miner_2_20280314_volscaled5_reversal_signal.csv')
obj={'factor_id':'miner_2_20280314_volscaled5_reversal_10d','factor_name':'Volatility-scaled five-day relative reversal','version':'1.0','calculation':{'expression':'-return_5d * clip(rolling_std(return_1d,5) / (rolling_std(return_1d,20)+epsilon), 0, 5)','description':'Contrarian five-session return scaled by the recent-to-medium volatility ratio; values are computed using data through the signal date only.'},'dependencies':['close'],'parameters':{'lookback_return_days':5,'short_vol_days':5,'medium_vol_days':20,'forward_horizon_days':10,'epsilon':1e-12,'vol_ratio_clip':[0,5]},'validation':{'status':'EFFECTIVE','period':'2020-01-01 through 2028-03-14','metrics':{'ic':0.03625479967251157,'icir':0.11117321686986166,'horizon_days':10,'dates':2441,'mean_instruments':15.0,'coverage':1.0,'turnover_proxy':0.31631031001355464,'hit_ratio':0.5460876689881197,'max_abs_library_correlation':None},'gate':{'min_abs_ic':0.007,'min_abs_icir':0.084},'regime_notes':'Positive in available 2025-2026 (IC 0.063953, ICIR 0.249441) and 2027-2028 (IC 0.037932, ICIR 0.133352); earlier files did not provide usable observations for this construction. Interpret small cross-section uncertainty conservatively.','signal_artifact':'scripts/miner_2_20280314_volscaled5_reversal_signal.csv'},'tags':['reversal','volatility','cross-asset','10d'],'last_validated':'2028-03-14T00:00:00Z'}
with open('factors/miner_2_20280314_volscaled5_reversal_10d.json','w') as f: json.dump(obj,f,indent=2)
print('written')
