import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for a in assets:
 p=os.path.join(base,a+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).set_index('date'); px[a]=d['close'].astype(float)
prices=pd.DataFrame(px).sort_index(); ret=prices.pct_change()
# observation-only VIX, lagged and aligned
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(prices.index).ffill()
vz=(v-v.rolling(60,min_periods=30).mean())/v.rolling(60,min_periods=30).std()
# lag all inputs: signal at t uses through t-1; reversal amplified only in high-vol regime
r5=prices.pct_change(5).shift(1); vol=ret.rolling(20,min_periods=15).std().shift(1)
reg=(1+0.35*np.tanh(vz.shift(1).fillna(0))).clip(.65,1.35)
f=(-r5/vol).mul(reg,axis=0)
# forward returns
outs=[]
for h in [1,5,10]:
 fr=prices.shift(-h)/prices-1
 ics=[]; ns=[]; dates=[]
 for dt in prices.index:
  x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 s=pd.Series(ics)
 outs.append((h,len(s),np.mean(ns),s.mean(),s.std(ddof=1) and s.mean()/s.std(ddof=1)*np.sqrt(252), (s>0).mean()))
# rank turnover
rnk=f.rank(axis=1,pct=True); turnover=(rnk.diff().abs().mean(axis=1)).dropna().mean()
print('dates',prices.index.min(),prices.index.max(),'assets',len(px),'rows',len(prices))
for x in outs: print('horizon dates avgN IC annualICIR hit',x)
print('coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',turnover)
# regimes by calendar
fr=prices.shift(-1)/prices-1
zlist=[]
for dt in prices.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:zlist.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(zlist,columns=['date','ic']).set_index('date')
for label,sub in q.groupby(pd.cut(q.index.year,[2019,2022,2024,2030],labels=['2020-22','2023-24','2025+'])): print(label,len(sub),sub.ic.mean())
q.to_csv('scripts/miner_1_20270218_vix_amplified_reversal_signal.csv')
