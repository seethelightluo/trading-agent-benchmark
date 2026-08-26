import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P0={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,900)
  except Exception: d=None
  if d is not None and len(d):break
 if d is not None and len(d):
  x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);P0[s]=x.drop_duplicates('date').set_index('date').close
P=pd.DataFrame(P0).sort_index();r=P.pct_change();sig=-(r.rolling(20,min_periods=15).std()*.7+(P/P.rolling(60,min_periods=40).max()-1)*-.3)
rows=[{'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(sig.loc[d,s])} for d in sig.index for s in sig.columns if pd.notna(sig.loc[d,s])];pd.DataFrame(rows).to_csv('scripts/miner_2_20290212_lowvol_drawdown_signal.csv',index=False)
def ev(h):
 v=[];co=[];to=[]
 for i in range(len(P)-h):
  z=pd.concat([sig.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));co.append(len(z)/15)
   if i:
    q=pd.concat([sig.iloc[i-1],sig.iloc[i]],axis=1).dropna()
    if len(q)>=8:to.append(1-q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 a=np.array(v);return len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),np.mean(co),np.nanmean(to)
for h in (5,10,20):print('H',h,'N IC ICIR hit cov turnover',ev(h))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028'),('2028','2029')]:
 z=[];lo=pd.Timestamp(a+'-01-01');hi=pd.Timestamp(b+'-12-31')
 for i in range(len(P)-10):
  if not(lo<=P.index[i]<=hi):continue
  q=pd.concat([sig.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print(a,b,len(z),np.nanmean(z),np.nanmean(z)/np.nanstd(z,ddof=1))
print('dates',P.index.min(),P.index.max(),'instruments',len(P.columns))
