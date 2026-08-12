import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT] for s in U}
for mn in ['VIX','USDJPY','EURUSD','USDCNY']:
 try: m=pd.read_csv('../persistent/index_data/'+mn+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT].close.pct_change()
 except Exception as e: print('missing',mn); continue
 fs={}; ys={}
 for s,x in D.items():
  r=x.close.pct_change(); cov=r.rolling(60,min_periods=30).cov(m); var=m.rolling(60,min_periods=30).var(); res=r-(cov/var)*m.reindex(r.index)
  fs[s]=-res.rolling(3,min_periods=3).sum()/(res.rolling(20,min_periods=15).std()*np.sqrt(3)+1e-12); ys[s]=x.close.shift(-1)/x.close-1
 out=[]; total=0; valid=0
 for dt in sorted(set().union(*[f.index for f in fs.values()])):
  g=pd.DataFrame({'f':{s:fs[s].get(dt,np.nan) for s in U},'y':{s:ys[s].get(dt,np.nan) for s in U}}).dropna(); total+=len(g)
  if len(g)>=8 and g.f.nunique()>1: out.append((dt,spearmanr(g.f,g.y).statistic,len(g))); valid+=len(g)
 z=pd.DataFrame(out,columns=['date','ic','n']); q=z.ic
 print(mn,'dates',len(q),'avgN',z.n.mean(),'coverage',valid/(len(set(z.date))*15),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  w=z[(z.date.dt.year>=lo)&(z.date.dt.year<=hi)].ic; print('reg',lo,len(w),w.mean(),w.mean()/w.std(ddof=1))
 for h in [5,10]:
  yy={s:D[s].close.shift(-h)/D[s].close-1 for s in U}; vals=[]
  for dt in sorted(set().union(*[f.index for f in fs.values()])):
   g=pd.DataFrame({'f':{s:fs[s].get(dt,np.nan) for s in U},'y':{s:yy[s].get(dt,np.nan) for s in U}}).dropna()
   if len(g)>=8 and g.f.nunique()>1: vals.append(spearmanr(g.f,g.y).statistic)
  v=np.array(vals); print('decay',h,'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'N',len(v))
