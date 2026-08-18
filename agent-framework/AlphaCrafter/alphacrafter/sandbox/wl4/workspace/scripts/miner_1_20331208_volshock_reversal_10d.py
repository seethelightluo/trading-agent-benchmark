import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
v10=r.rolling(10,min_periods=8).std(); v60=r.rolling(60,min_periods=30).std()
# contrarian return emphasized after volatility shock, normalized by medium-term risk
sig=-(px/px.shift(10)-1)*(v10/v60)/(v60*np.sqrt(10))
ics=[]; ns=[]; cov=[]; turns=[]; prev=None
for d in sig.index:
 x=sig.loc[d]; y=(px.shift(-10)/px-1).loc[d]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  q=spearmanr(x[ok],y[ok]).statistic;ics.append(q);ns.append(ok.sum());cov.append(ok.mean())
  if prev is not None: turns.append((x[ok].rank()-prev[ok].rank()).abs().sum()/(len(U)**2))
  prev=x
z=np.asarray(ics);print('factor=volshock_reversal_10d dates',len(z),'avg_inst',np.mean(ns),'coverage',np.mean(cov),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turns))
for n in [260,520,780]:
 q=z[-n:];print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'dates',len(q))
for h in [5,10,20]:
 yy=px.shift(-h)/px-1;a=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8:a.append(spearmanr(sig.loc[d][ok],yy.loc[d][ok]).statistic)
 q=np.asarray(a);print('decay',h,q.mean(),q.mean()/q.std(ddof=1),len(q))
