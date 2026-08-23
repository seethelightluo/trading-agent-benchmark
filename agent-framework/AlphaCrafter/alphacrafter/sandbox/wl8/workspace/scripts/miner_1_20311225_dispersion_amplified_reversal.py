import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2031-12-25')
P=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.rename(a) for a in A],axis=1).sort_index().loc[:E]
R=np.log(P).diff(); mom=R.rolling(5,min_periods=4).sum().shift(1); disp=R.T.rolling(10,min_periods=8).std().T.shift(1); base=disp.rolling(252,min_periods=63).median().shift(1)
g=(disp/base).clip(.5,2.5); F=-mom.mul(g,axis=0)
fr=np.log(P.shift(-10)/P); out=[]; ns=[]; ds=[]
for d in F.index:
 z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):out.append(q);ns.append(len(z));ds.append(d)
a=np.array(out); print({'dates':len(a),'start':str(ds[0].date()),'end':str(ds[-1].date()),'avg_names':round(np.mean(ns),2),'coverage':round(F.notna().mean().mean(),4),'ic':round(a.mean(),6),'icir':round(a.mean()/a.std(ddof=1),6),'hit':round((a>0).mean(),4),'turnover':round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6)})
for n in [180,365,756]:
 b=a[-n:]; print('recent',n,round(b.mean(),6),round(b.mean()/b.std(ddof=1),6),len(b))
for h in [5,10,20]:
 ff=np.log(P.shift(-h)/P); x=[]
 for d in F.index:
  z=pd.concat([F.loc[d],ff.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x);print('decay',h,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),len(x))
pd.DataFrame({'date':ds,'ic':a,'n':ns}).to_csv('scripts/miner_1_20311225_dispersion_amplified_reversal_ic.csv',index=False);F.to_csv('scripts/miner_1_20311225_dispersion_amplified_reversal_signal.csv')
