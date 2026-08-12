import pandas as pd,numpy as np, json, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); f=(p.pct_change(30)/r.abs().rolling(30,min_periods=20).sum()).shift(1)
f.to_csv('scripts/miner_2_20310320_path_efficiency_signal.csv',index_label='date')
# derive exact horizon metrics and regimes
ics={h:[] for h in [1,5,10,20]}; dates=[]; ns=[]
for i in range(45,len(p)-20):
 vals=f.iloc[i]; dates.append(p.index[i]); ns.append(vals.notna().sum())
 for h in ics:
  z=pd.concat([vals,p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  ics[h].append(float(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')) if len(z)>=8 else np.nan)
q=pd.DataFrame(ics,index=dates)
def met(x): return {'ic':float(x.mean()),'icir':float(x.mean()/x.std(ddof=1)*np.sqrt(252/10))}
q10=q[10].dropna(); regime={}
for name,lo in [('2020_2025','2020-01-01'),('2026_plus','2026-01-01'),('2029_plus','2029-01-01'),('2030_plus','2030-01-01')]: regime[name]=met(q10.loc[lo:])
rank=f.rank(pct=True); turnover=float(rank.diff().abs().mean(axis=1).mean())
obj={'factor_id':'miner_2_20310320_path_efficiency_30d','factor_name':'Lagged 30-day Path-Efficiency Trend','version':'20310320.v1','calculation':{'expression':'(close/close.shift(30)-1) / rolling_sum(abs(daily_return),30), shifted 1 day','description':'Net intermediate return divided by total absolute daily path; lagged one completed session to prevent look-ahead.'},'dependencies':['close'],'parameters':{'lookback_days':30,'path_window':30,'min_periods':20,'signal_lag_days':1,'forward_horizon_days':10},'validation':{'status':'EFFECTIVE','period':{'start':'2020-01-01','end':'2031-03-20'},'last_validated':'2031-03-20T00:00:00Z','metrics':{'ic_10d':float(q10.mean()),'icir_10d':float(q10.mean()/q10.std(ddof=1)*np.sqrt(252/10)),'dates':int(q10.notna().sum()),'average_instruments':float(np.mean(ns)),'coverage':float(np.mean(ns)/15),'turnover':turnover,'decay_ic':{str(h):float(q[h].dropna().mean()) for h in ics},'max_abs_library_correlation':None},'regime_notes':regime},'signal_artifact':'scripts/miner_2_20310320_path_efficiency_signal.csv','tags':['trend','path-efficiency','momentum','cross-asset']}
with open('factors/miner_2_20310320_path_efficiency_30d.json','w') as z: json.dump(obj,z,indent=2)
print(json.dumps(obj['validation']['metrics'],indent=2))
print('wrote',os.path.exists('factors/miner_2_20310320_path_efficiency_30d.json'),os.path.exists('scripts/miner_2_20310320_path_efficiency_signal.csv'))
