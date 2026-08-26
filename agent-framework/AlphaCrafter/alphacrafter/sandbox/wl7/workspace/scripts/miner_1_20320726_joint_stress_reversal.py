import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d): return d.set_index('date')['close']
  except: pass
p={s:get(s) for s in U}; p=pd.DataFrame(p).sort_index()
macro={}
for s in ['DXY','VIX']:
 d=pd.read_csv('../persistent/index_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']; macro[s]=d
m=pd.DataFrame(macro).reindex(p.index).ffill()
# only information through t; signal: high joint stress -> short 5d reversal, normal -> 20d momentum
r=p.pct_change(); v=r.rolling(20,min_periods=15).std();
dxy=m.DXY.pct_change(20); vix5=m.VIX.pct_change(5)
stress=(dxy>0)&(vix5>0)
f=np.where(stress.values[:,None], -r.rolling(5,min_periods=5).sum().values/v.values, r.rolling(20,min_periods=15).sum().values/v.values)
f=pd.DataFrame(f,index=p.index,columns=p.columns)
# lag signal one date relative to forward returns
out=[]
for h in [1,5,10,20]:
 vals=[]
 for i in range(len(p)-h-1):
  dt=p.index[i]; x=f.iloc[i]; y=p.iloc[i+1:i+h+1].iloc[-1]/p.iloc[i]-1
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(vals); out.append((h,len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
valid=f.notna().any(axis=1); ranks=f.rank(axis=1,pct=True); turn=(ranks.diff().abs().mean(axis=1)>0.15).mean()
print('cutoff',p.index[-1].date(),'calendar',len(p),'valid_dates',valid.sum(),'avgN',f.notna().sum(axis=1)[valid].mean(),'coverage',f.notna().mean().mean(),'turnover',turn)
for x in out: print('H',x[0],'dates',x[1],'IC %.8f ICIR %.8f hit %.3f'%x[2:])
# chronological H10 thirds
h=10; vals=[]; dates=[]
for i in range(len(p)-h-1):
 z=pd.concat([f.iloc[i],p.iloc[i+1:i+h+1].iloc[-1]/p.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]));dates.append(p.index[i])
a=np.array(vals)
for q in np.array_split(a,3): print('third',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
# artifact
sig=f.loc[:'2032-07-25']; sig.to_csv('scripts/miner_1_20320726_joint_stress_reversal_signal.csv')
