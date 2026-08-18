import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
# 30-day trend, risk-normalized by trailing 30-day realized volatility
sig=(px/px.shift(30)-1)/(r.rolling(30,min_periods=20).std()*np.sqrt(30))
ics=[]; ns=[]; cov=[]; turns=[]; prev=None
for i,d in enumerate(sig.index):
 x=sig.loc[d]; y=(px.shift(-10)/px-1).loc[d]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  q=spearmanr(x[ok],y[ok]).statistic; ics.append(q); ns.append(ok.sum()); cov.append(ok.mean())
  if prev is not None:
   turns.append((x[ok].rank()-prev[ok].rank()).abs().sum()/(len(U)**2))
  prev=x
z=np.asarray(ics); print('factor=volnorm_momentum_30d dates',len(z),'avg_inst',np.mean(ns),'coverage',np.mean(cov),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turns))
for n in [260,520,780]:
 q=z[-n:]; print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'dates',len(q))
for h in [5,10,20]:
 yy=px.shift(-h)/px-1; a=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8:a.append(spearmanr(sig.loc[d][ok],yy.loc[d][ok]).statistic)
 q=np.asarray(a); print('decay',h,q.mean(),q.mean()/q.std(ddof=1),len(q))
