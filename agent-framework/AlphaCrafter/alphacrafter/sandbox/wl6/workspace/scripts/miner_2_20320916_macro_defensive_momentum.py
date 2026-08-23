import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2032-09-15'); base=Path('../persistent/stock_data'); macro=Path('../persistent/index_data')
px={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
wide=pd.DataFrame(px).sort_index().loc[lambda x:x.index<=cutoff]; ret=wide.pct_change()
vix=pd.read_csv(macro/'VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(wide.index).ffill()
# VIX-premium penalty: risk-adjusted 40d momentum is discounted only when volatility is unusually elevated.
shock=(vix/vix.rolling(120,min_periods=80).median()-1).clip(lower=0,upper=2)
vol=ret.rolling(40,min_periods=30).std(); factor=wide.pct_change(40)/(vol*np.sqrt(40)+1e-12); factor=factor.div(1+shock,axis=0)
for h in [5,10,20]:
 a=[]; ns=[]
 for i in range(len(wide)-h):
  z=pd.concat([factor.iloc[i],wide.iloc[i+h]/wide.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),3),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round(np.mean(a>0),6))
print('coverage',round(factor.notna().mean().mean(),6),'turnover',round(factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),8),'cutoff',cutoff.date())
for yr in range(2026,2033):
 a=[]
 for i in range(len(wide)-10):
  if wide.index[i].year!=yr: continue
  z=pd.concat([factor.iloc[i],wide.iloc[i+10]/wide.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('year',yr,'n',len(a),'ic10',round(float(np.mean(a)),8) if a else None)
