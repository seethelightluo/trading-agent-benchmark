import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
px=pd.concat({s:D[s]['close'] for s in U},axis=1).sort_index(); hi=pd.concat({s:D[s]['high'] for s in U},axis=1).reindex(px.index); lo=pd.concat({s:D[s]['low'] for s in U},axis=1).reindex(px.index)
r=px.pct_change(); clv=((px-lo)/(hi-lo+1e-12)-.5); rng=(hi-lo)/px
# interpretable range-expansion pressure: CLV sign/magnitude weighted by today's range relative to its 20d median
exp=rng/rng.rolling(20,min_periods=15).median(); f=clv* np.log1p(exp.clip(lower=0))
for h in [1,5,10]:
 fut=px.pct_change(h).shift(-h); out=[]; ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],fut.iloc[i]],axis=1).dropna()
  if len(z)>=8:out.append(z.iloc[:,0].corr(z.iloc[:,1]));ns.append(len(z))
 a=np.array(out); print('h',h,'dates',len(a),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
# rank turnover
q=f.rank(axis=1,pct=True); print('turnover',q.diff().abs().mean(axis=1).mean())
