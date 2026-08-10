import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data/'
C=pd.concat({s:pd.read_csv(P+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index()
# 20-session reversal with 5-session skip; all inputs lagged at decision t
r20=C/C.shift(25)-1
f=-r20.shift(1)
# forward 10 sessions from t, no look-ahead
fw=C.shift(-10)/C-1
ics=[]; ns=[]; turns=[]; prev=None; dates=[]
for d in C.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic; ics.append(ic);ns.append(len(z));dates.append(d)
  rank=z.iloc[:,0].rank(pct=True)
  if prev is not None: turns.append(np.abs(rank-prev).mean())
  prev=rank
A=np.array(ics); print('factor 20d reversal skip5; dates',len(A),'avgN',np.mean(ns),'IC',A.mean(),'ICIR',A.mean()/A.std(ddof=1),'hit',(A>0).mean(),'turn',np.mean(turns),'coverage',f.notna().sum().sum()/f.size)
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 a=np.array([x for x,d in zip(ics,dates) if lo<=str(d.year)<=hi]);print(lo,hi,'n',len(a),'IC',a.mean() if len(a) else np.nan,'ICIR',a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
for h in [1,5,20]:
 fw=C.shift(-h)/C-1;a=[]
 for d in C.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print('decay',h,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('../persistent/factor_signals_miner_1_20270225_reversal20_skip5.csv')
