import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-11-04'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
R=P.pct_change(); atr=(P.diff().abs()/P.shift(1)).shift(1).rolling(20,min_periods=12).mean()
# Mean reversion in units of average absolute daily move, lagged one completed day.
sig=-(P.shift(1)/P.shift(11)-1)/atr
sig=sig.replace([np.inf,-np.inf],np.nan)
print('rows',len(P),'assets',P.notna().sum().min())
for h in [5,10,20,40]:
 f=P.shift(-h)/P-1; out=[]; ns=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt].rename('x'),f.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1: out.append(spearmanr(z.x,z.y).statistic);ns.append(len(z))
 q=pd.Series(out).dropna(); print('H',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
f=P.shift(-20)/P-1
for lab,mask in [('early',P.index<='2026-12-31'),('mid',(P.index>='2027-01-01')&(P.index<='2028-12-31')),('late',P.index>='2029-01-01')]:
 q=[]
 for dt in P.index[mask]:
  z=pd.concat([sig.loc[dt].rename('x'),f.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1:q.append(spearmanr(z.x,z.y).statistic)
 q=pd.Series(q).dropna(); print(lab,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
rank=sig.rank(axis=1,pct=True); t=[]
for a,b in zip(rank.index[:-1],rank.index[1:]):
 z=pd.concat([rank.loc[a],rank.loc[b]],axis=1).dropna()
 if len(z):t.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
print('turnover',round(float(np.mean(t)),6),'valid_dates',sig.dropna(how='all').shape[0])
sig.to_csv('scripts/miner_3_20291105_atr_reversal10_signal.csv')
