import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for a in assets:
 p=os.path.join(base,a+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).set_index('date'); px[a]=d.close.astype(float)
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change()
# Candidate: lagged 10-day momentum normalized by recent volatility and confirmed by 5/20-day direction agreement.
r10=prices.pct_change(10).shift(1); vol=ret.rolling(20,min_periods=15).std().shift(1)
r5=prices.pct_change(5).shift(1); r20=prices.pct_change(20).shift(1)
confirm=((np.sign(r5)==np.sign(r10)).astype(float)+(np.sign(r20)==np.sign(r10)).astype(float))/2
f=(r10/vol)*(.5+.5*confirm)
results=[]; allics=[]
for h in [1,5,10,20]:
 fr=prices.shift(-h)/prices-1; ics=[]; ns=[]; dates=[]
 for dt in prices.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 s=pd.Series(ics); results.append((h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1)*np.sqrt(252),float((s>0).mean())))
 if h==1: allics=pd.Series(ics,index=dates)
rnk=f.rank(axis=1,pct=True); turnover=rnk.diff().abs().mean(axis=1).dropna().mean()
print('period',prices.index.min().date(),prices.index.max().date(),'assets',len(px),'rows',len(prices))
for x in results: print('horizon dates avgN IC annualICIR hit',x)
print('coverage_dates_ge8',f.notna().sum(axis=1).ge(8).mean(),'mean_valid_assets',f.notna().sum(axis=1).mean(),'turnover',turnover)
for label,sub in allics.groupby(pd.cut(allics.index.year,[2019,2022,2024,2030],labels=['2020-22','2023-24','2025+'])): print('regime',label,'dates',len(sub),'IC',sub.mean())
pd.DataFrame({'date':allics.index,'signal_ic':allics.values}).to_csv('scripts/miner_1_20270317_trend_agreement_signal.csv',index=False)
