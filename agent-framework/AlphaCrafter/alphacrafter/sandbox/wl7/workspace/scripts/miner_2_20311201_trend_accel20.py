import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=5000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index().ffill()
# candidate: trend acceleration, medium trend normalized by slow volatility, relative to universe median
r=P.pct_change()
base=r.rolling(20).sum()/r.rolling(60).std()
acc=base-base.shift(10)
# lag signal one day, cross-sectional demeaning is rank-neutral
sig=acc.shift(1)
rows=[]
for h in [1,5,10,20]:
  vals=[]
  for i in range(len(P)-h):
    dt=P.index[i]; nxt=P.iloc[i+h]/P.iloc[i]-1
    a=sig.iloc[i]
    z=pd.concat([a,nxt],axis=1).dropna()
    if len(z)>=8:
      vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
  q=pd.DataFrame(vals,columns=['date','ic','n'])
  m=q.ic.mean(); sd=q.ic.std(ddof=1)
  print(h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd*np.sqrt(252),4),'hit',round((q.ic>0).mean(),4))
# coverage and turnover of rank signal
valid=sig.notna().sum().sum()/(len(sig)*len(U))
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('coverage',round(valid,4),'turnover',round(turn,6),'period',P.index.min().date(),P.index.max().date(),'assets',len(D))
# artifact for best horizon
h=20; out=[]
ics=[]
for i in range(len(P)-h):
 dt=P.index[i]; a=sig.iloc[i]; nxt=P.iloc[i+h]/P.iloc[i]-1
 z=pd.concat([a,nxt],axis=1).dropna()
 if len(z)>=8: ics.append({'date':dt,'ic':z.iloc[:,0].corr(z.iloc[:,1]),'n':len(z)})
pd.DataFrame(ics).to_csv('scripts/miner_2_20311201_trend_accel20_ic.csv',index=False)
for i in range(len(P)-h):
 dt=P.index[i]; a=sig.iloc[i]; nxt=P.iloc[i+h]/P.iloc[i]-1
 for s in U:
  if pd.notna(a.get(s)) and pd.notna(nxt.get(s)): out.append({'date':dt,'symbol':s,'signal':a[s],'forward_return':nxt[s]})
pd.DataFrame(out).to_csv('scripts/miner_2_20311201_trend_accel20_signal.csv',index=False)
