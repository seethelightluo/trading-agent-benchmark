import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; X={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): X[a]=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(X).sort_index(); r=P.pct_change(); down=r.where(r<0,0).pow(2).rolling(40,min_periods=20).mean().pow(.5); F=(-P.pct_change(10)/(down+1e-8)).shift(1); rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],P.shift(-10).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
s=pd.Series(rows); print('dates',len(s),'assets',len(P.columns),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean(),'coverage',F.notna().sum(axis=1).mean()/len(A),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
for n in [260,520,780]:
 q=s.tail(n); print('recent',n,q.mean(),q.mean()/q.std(),(q>0).mean())
os.makedirs('scripts/artifacts',exist_ok=True); F.to_csv('scripts/artifacts/miner_2_20330818_downside_risk_reversal10_signal.csv')
