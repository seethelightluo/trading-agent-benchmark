import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-12-24')
D={}
for a in assets:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date'); D[a]=x.loc[x.index<=cutoff]
prices=pd.concat({a:D[a]['close'] for a in D},axis=1).sort_index(); r=prices.pct_change()
f=-(prices/prices.shift(20)-1)/(r.abs().rolling(20).sum()+1e-8)
for h in [5,10,20]:
  ics=[]; ns=[]; turns=[]
  for i in range(20,len(prices)-h):
   z=f.iloc[i]; y=prices.iloc[i+h]/prices.iloc[i]-1; ok=z.notna()&y.notna()
   if ok.sum()>=8: ics.append(spearmanr(z[ok],y[ok]).statistic); ns.append(ok.sum())
   if i>20:
    old=f.iloc[i-1]; oo=old.notna()&z.notna()
    if oo.sum(): turns.append(np.mean((z[oo].rank(pct=True)-old[oo].rank(pct=True)).abs()))
  a=np.array(ics); print('horizon dates avgN IC ICIR hit turnover',h,len(a),round(np.mean(ns),2),round(np.mean(a),6),round(np.mean(a)/np.std(a,ddof=1),4),round(np.mean(a>0),4),round(np.mean(turns),6))
print('assets',len(D),'dates',len(prices),'coverage',round(f.notna().mean().mean(),6),'cutoff',prices.index.max().date())
