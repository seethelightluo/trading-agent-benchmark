import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
END='2027-02-25'
files=glob.glob('../persistent/stock_data/*.csv')
prices={}
for f in files:
 s=os.path.basename(f)[:-4]; d=pd.read_csv(f); d['date']=pd.to_datetime(d['date']); d=d[d.date<=END].sort_values('date')
 if len(d)>100: prices[s]=d.set_index('date')
assets=sorted(prices); dates=sorted(set.intersection(*[set(x.index) for x in prices.values()]))
# Range-expansion-conditioned one-day reversal: yesterday return, activated if yesterday's range is unusually large
rows=[]
for i in range(60,len(dates)-10):
 dt=dates[i]; prev=dates[i-1]
 sig={}; fwd={}
 for a in assets:
  x=prices[a]
  if prev not in x.index or dt not in x.index: continue
  hist=x.loc[:prev].tail(60)
  tr=((hist.high-hist.low)/hist.close).iloc[-1]
  q=((hist.high-hist.low)/hist.close).rolling(60).quantile(.75).iloc[-1]
  if pd.notna(tr) and pd.notna(q) and tr>=q:
   r=x.close.pct_change().loc[prev]
   sig[a]=-r
   j=dates.index(dt); end=dates[j+5] if j+5<len(dates) else None
   if end in x.index: fwd[a]=x.close.loc[end]/x.close.loc[prev]-1
 if len(sig)>=8:
  for h in [1,5,10]:
   vals=[]
   j=dates.index(dt)
   for a,v in sig.items():
    k=j+h
    if k<len(dates) and dates[k] in prices[a].index:
     vals.append((v,prices[a].close.loc[dates[k]]/prices[a].close.loc[prev]-1))
   if len(vals)>=8:
    ic=spearmanr(*zip(*vals)).statistic
    rows.append((dt,h,ic,len(vals)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
print('assets',len(assets),'dates',len(dates),'active observations',df.date.nunique())
for h in [1,5,10]:
 z=df[df.h==h].ic.dropna(); print(h,'obs',len(z),'meanIC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'avgN',df[df.h==h].n.mean())
# artifact daily signals
out=[]
for i in range(60,len(dates)):
 dt=dates[i]; prev=dates[i-1]
 for a in assets:
  x=prices[a]; hist=x.loc[:prev].tail(60)
  tr=((hist.high-hist.low)/hist.close).iloc[-1]; q=((hist.high-hist.low)/hist.close).rolling(60).quantile(.75).iloc[-1]
  if tr>=q: out.append({'date':dt,'asset':a,'signal':-x.close.pct_change().loc[prev]})
os.makedirs('../persistent',exist_ok=True); pd.DataFrame(out).to_csv('../persistent/factor_signals_miner_3_20270225_rangeexp_reversal.csv',index=False)
