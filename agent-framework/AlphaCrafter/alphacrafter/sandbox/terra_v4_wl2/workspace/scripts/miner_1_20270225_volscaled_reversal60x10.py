import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P='../persistent/stock_data/'
C=pd.concat({s:pd.read_csv(P+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index(); r=C.pct_change()
# medium reversal scaled by trailing volatility, all lagged one day
raw=-(C.shift(11)/C.shift(71)-1); vol=r.rolling(20).std().shift(1); f=raw/vol
fw=C.shift(-10)/C-1; out=[];ns=[];ds=[];turn=[];prev=None
for d in C.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8:
  out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(d)
  q=z.iloc[:,0].rank(pct=True)
  if prev is not None:turn.append(abs(q-prev).mean())
  prev=q
A=np.array(out);print('volscaled reversal 60x10 h10 dates',len(A),'avgN',np.mean(ns),'IC',A.mean(),'ICIR',A.mean()/A.std(ddof=1),'hit',(A>0).mean(),'turn',np.mean(turn),'coverage',f.notna().sum().sum()/f.size)
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 a=np.array([x for x,d in zip(out,ds) if lo<=str(d.year)<=hi]);print(lo,hi,len(a),a.mean() if len(a) else np.nan,a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
for h in [1,5,20]:
 y=C.shift(-h)/C-1;a=[]
 for d in C.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print('decay',h,len(a),a.mean(),a.mean()/a.std(ddof=1))
X=f.copy();X.index=X.index.strftime('%Y-%m-%d');X.to_csv('../persistent/factor_signals_miner_1_20270225_volscaled_reversal60x10.csv')
