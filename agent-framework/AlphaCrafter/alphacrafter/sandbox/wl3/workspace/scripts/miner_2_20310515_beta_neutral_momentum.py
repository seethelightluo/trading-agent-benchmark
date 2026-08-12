import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s, days=5000)
 if d is not None and len(d): raw[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(raw).sort_index().ffill()
# Candidate: beta-neutral medium momentum. Remove each asset's rolling beta to equal-weight benchmark over 60d,
# then scale residual 20d return by residual volatility. Factor is lagged one session.
r=np.log(p).diff(); bench=r.mean(axis=1)
out=[]
for t in range(80,len(p)-10):
 dt=p.index[t]
 rr=r.iloc[:t+1]; b=bench.iloc[:t+1]
 vals={}
 for s in U:
  x=rr[s].iloc[-60:]; y=b.iloc[-60:]
  ok=x.notna()&y.notna()
  if ok.sum()<40: continue
  beta=np.cov(x[ok],y[ok],ddof=1)[0,1]/(np.var(y[ok],ddof=1)+1e-12)
  resid=x-beta*y
  vals[s]=resid.iloc[-20:].sum()/(resid.iloc[-40:].std()+1e-8)/np.sqrt(20)
 f=pd.Series(vals)
 # factor direction: trend continuation
 fr=(p.iloc[t+10]/p.iloc[t]-1)
 z=pd.concat([f,fr],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  out.append((dt,ic,len(z),f))
ics=pd.Series({x[0]:x[1] for x in out}).dropna()
print('dates',len(ics),'avg_n',np.mean([x[2] for x in out]),'IC',ics.mean(),'ICIR',ics.mean()/(ics.std(ddof=1)+1e-12)*np.sqrt(len(ics)),'hit',(ics>0).mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=ics[(ics.index.year>=int(a))&(ics.index.year<=int(b))]
 print(a,b,len(q),q.mean(),q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(len(q)) if len(q)>1 else np.nan)
# turnover of rank signal between dates
fs=[x[3] for x in out]; turns=[]
for x,y in zip(fs[:-1],fs[1:]):
 z=pd.concat([x.rank(),y.rank()],axis=1).dropna(); turns.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean()/(len(z)))
print('coverage',sum(x[2] for x in out)/(len(out)*15),'turnover',np.mean(turns))
# save artifact usable provenance
rows=[]
for dt,ic,n,f in out:
 for s,v in f.items(): rows.append({'date':dt,'symbol':s,'signal':v})
pd.DataFrame(rows).to_csv('scripts/miner_2_20310515_beta_neutral_momentum_signal.csv',index=False)
