import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): raw[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(raw).sort_index().ffill(); r=np.log(p).diff(); med=r.median(axis=1)
# Observation-only VIX: high-volatility state, all values lagged through date t.
v=pd.read_csv('../persistent/index_data/VIX.csv')
v['date']=pd.to_datetime(v['date']); v=v.set_index('date')['close'].astype(float).reindex(p.index).ffill()
out=[]
for t in range(260,len(p)-10):
 if pd.isna(v.iloc[t]): continue
 # activated only after VIX has been above its trailing 252-session median on two sessions
 vm=v.iloc[t-252:t+1]
 active=(v.iloc[t]>vm.median()) and (v.iloc[t-1]>v.iloc[t-253:t].median())
 if not active: continue
 vals={}
 for s in U:
  q=(r[s]-med).iloc[:t+1]
  if q.iloc[-45:].notna().sum()<35: continue
  vol=q.iloc[-20:].std()
  vals[s]=-q.iloc[-5:].sum()/(vol*np.sqrt(5)+1e-8)
 f=pd.Series(vals); fr=p.iloc[t+10]/p.iloc[t]-1
 z=pd.concat([f,fr],axis=1).dropna()
 if len(z)>=8: out.append((p.index[t],z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z),f))
ics=pd.Series({x[0]:x[1] for x in out}).dropna()
print('dates',len(ics),'avg_n',np.mean([x[2] for x in out]),'IC',ics.mean(),'ICIR',ics.mean()/(ics.std(ddof=1)+1e-12)*np.sqrt(len(ics)),'hit',(ics>0).mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=ics[(ics.index.year>=int(a))&(ics.index.year<=int(b))]; print(a,b,len(q),q.mean(),q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(len(q)) if len(q)>1 else np.nan)
fs=[x[3] for x in out]; turns=[]
for x,y in zip(fs[:-1],fs[1:]):
 z=pd.concat([x.rank(),y.rank()],axis=1).dropna(); turns.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean()/len(z))
print('coverage',sum(x[2] for x in out)/(len(out)*15),'turnover',np.mean(turns))
rows=[]
for dt,ic,n,f in out:
 for s,val in f.items(): rows.append({'date':dt,'symbol':s,'signal':val})
pd.DataFrame(rows).to_csv('scripts/miner_2_20310529_highvix_residual_reversal_signal.csv',index=False)
