import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-05-13')
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets}
p=pd.DataFrame(p).sort_index(); p=p[p.index<=cut]; r=p.pct_change()
# Broader dispersion-conditioned residual reversal: use 5d relative loss, volatility normalize,
# active above the trailing 252d median of 20d cross-asset dispersion (lagged state).
disp=r.std(axis=1).rolling(20).mean(); threshold=disp.rolling(252).quantile(.50)
active=(disp>threshold).shift(1).fillna(False)
raw=-(p.pct_change(5).sub(p.pct_change(5).median(axis=1),axis=0))
f=(raw/r.rolling(20).std()).where(active)
for h in [1,3,5,10]:
 fr=p.pct_change(h).shift(-h); ics=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(ics); print(h,'dates',len(a),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
for s,e in [('2026-01-01','2029-12-31'),('2030-01-01','2033-05-13')]:
 a=[]
 for dt in f.index:
  if not(pd.Timestamp(s)<=dt<=pd.Timestamp(e)): continue
  z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(a);print('regime',s,'dates',len(a),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
