import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
px=pd.DataFrame(P).sort_index(); ret=px.pct_change();
# interpretable signal: contrarian 20d return scaled by trailing 20d realized volatility
sig=-(px/px.shift(20)-1)/(ret.rolling(20,min_periods=15).std()*np.sqrt(20))
fwd=px.shift(-10)/px-1
ics=[]; turnovers=[]; cov=[]; ninst=[]
for d in sig.index:
 x=sig.loc[d]; y=fwd.loc[d]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic); ninst.append(ok.sum()); cov.append(ok.mean())
  if d in sig.index and sig.index.get_loc(d)>0:
   prev=sig.iloc[sig.index.get_loc(d)-1]; a=x.rank(); b=prev.rank(); turnovers.append((a-b).abs().sum()/(len(U)**2))
a=np.array(ics); print('factor=volnorm_reversal_20d dates',len(a),'avg_inst',np.mean(ninst),'coverage',np.mean(cov),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',np.mean(turnovers))
for n in [260,520,780]:
 z=a[-n:]; print('recent',n,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'dates',len(z))
for h in [5,10,20]:
 yy=px.shift(-h)/px-1; aa=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8: aa.append(spearmanr(sig.loc[d][ok],yy.loc[d][ok]).statistic)
 z=np.array(aa); print('decay',h,z.mean(),z.mean()/z.std(ddof=1),len(z))
