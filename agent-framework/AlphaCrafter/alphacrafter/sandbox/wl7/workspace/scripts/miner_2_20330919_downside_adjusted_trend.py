import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date)
 D[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); ret=px.pct_change()
# Lagged downside-adjusted medium-term trend: 20-session return divided by
# downside deviation over the preceding 40 sessions; all inputs end at t-1.
lag=px.shift(1); mom=lag.pct_change(20)
dr=ret.shift(1).where(ret.shift(1)<0,0.0).rolling(40).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True)
f=(mom/dr).replace([np.inf,-np.inf],np.nan)
print('instruments',len(D),'dates',len(px),'coverage',f.notna().mean().mean())
for h in [1,5,10,20]:
 a=[]; ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:a.append(z.x.corr(z.y));ns.append(len(z))
 q=pd.Series(a).dropna(); print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'n',len(q),'avgN',np.mean(ns))
q=[]
for i in range(len(px)-10):
 z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+10]/px.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8 and z.x.nunique()>2:q.append((px.index[i],z.x.corr(z.y)))
q=pd.Series(dict(q));print('thirds H10',*[q.iloc[j*len(q)//3:(j+1)*len(q)//3].mean() for j in range(3)])
rank=f.rank(axis=1,pct=True);print('turnover',rank.diff().abs().mean().mean())
f.index=f.index.astype(str);f.to_csv('scripts/miner_2_20330919_downside_adjusted_trend_signal.csv')
