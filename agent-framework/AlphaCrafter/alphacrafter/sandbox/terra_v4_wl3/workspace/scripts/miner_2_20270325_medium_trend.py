import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-03-24')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:end]
 px[s]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# medium horizon risk-adjusted trend, all information through t
sig=(p.pct_change(10)/r.rolling(20).std()).replace([np.inf,-np.inf],np.nan)
# forward returns and lag signal naturally date t predicts t+1
rows=[]
for t in sig.index:
 if t not in r.index: continue
 f=r.shift(-1).loc[t]
 x=sig.loc[t]
 ok=x.notna()&f.notna()
 if ok.sum()>=8:
  rows.append((t,spearmanr(x[ok],f[ok]).statistic,ok.sum()))
df=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(df),'avgN',df.n.mean(),'coverage',df.n.sum()/(len(df)*15))
print('daily mean/ICIR/hit',df.ic.mean(),df.ic.mean()/df.ic.std(),(df.ic>0).mean())
for h in [5,10]:
 f=p.pct_change(h).shift(-h); arr=[]
 for t in sig.index:
  x=sig.loc[t]; y=f.loc[t]; ok=x.notna()&y.notna()
  if ok.sum()>=8: arr.append(spearmanr(x[ok],y[ok]).statistic)
 a=np.array(arr);print(h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean())
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-24')]:
 z=df.loc[a:b].ic;print(a,b,len(z),z.mean(),z.mean()/z.std() if len(z)>1 else np.nan)
# rank turnover top/bottom ordering changes
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean();print('turnover proxy',turn)
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270325_medium_trend_signal.csv',index=False)
