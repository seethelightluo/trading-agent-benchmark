import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2029-06-13')
fs={}
for a in assets:
 p=f'../persistent/stock_data/{a}.csv'; d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
 fs[a]=d['close'].replace(0,np.nan)
px=pd.concat(fs,axis=1).sort_index().loc[:end].ffill()
# Volatility-scaled medium-term reversal: negative 60d return / 20d realized volatility.
sig=-px.pct_change(60)/(px.pct_change().rolling(20).std()*np.sqrt(252)); fw=px.shift(-10)/px-1
ics=[]; dates=[]; ns=[]
for d in sig.index:
 ok=sig.loc[d].notna()&fw.loc[d].notna()
 if ok.sum()>=8: ics.append(spearmanr(sig.loc[d][ok],fw.loc[d][ok]).statistic); dates.append(d); ns.append(ok.sum())
z=np.array(ics); print('period',dates[0].date(),dates[-1].date(),'valid_dates',len(z),'mean_n',np.mean(ns),'coverage',np.mean(ns)/15)
print('full IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
for c in ['2026-07-16','2027-01-01','2028-06-14','2029-01-01']:
 q=z[np.array(dates)>=pd.Timestamp(c)]; print(c,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for h in [1,5,10,20,40]:
 fw=px.shift(-h)/px-1; q=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&fw.loc[d].notna()
  if ok.sum()>=8:q.append(spearmanr(sig.loc[d][ok],fw.loc[d][ok]).statistic)
 q=np.array(q); print('horizon',h,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
