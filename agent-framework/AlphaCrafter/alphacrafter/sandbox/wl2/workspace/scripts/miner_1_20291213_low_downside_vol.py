import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date') for s in U}
# aligned panel, factor uses close through date: downside-adjusted low-volatility score
cl=pd.DataFrame({s:d.close for s,d in D.items()}); ret=cl.pct_change()
# factor: prefer assets with low vol and shallow downside, cross-section rank blended
vol=ret.rolling(30,min_periods=20).std()
down=ret.clip(upper=0).rolling(30,min_periods=20).std()
# high score = low total vol and low downside volatility, winsorized rank
f=-(0.6*vol.rank(axis=1,pct=True)+0.4*down.rank(axis=1,pct=True))
rows=[]
for i,dt in enumerate(cl.index[:-10]):
    x=f.loc[dt];
    if i+1>=len(cl.index): continue
    y=ret.shift(-1).loc[dt]
    ok=x.notna()&y.notna()
    if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum(),x[ok].nunique()))
a=pd.DataFrame(rows,columns=['date','ic','n','nu']).set_index('date')
print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
for h in [1,5,10]:
    y=ret.shift(-h); rs=[]
    for dt in f.index:
      x=f.loc[dt]; z=y.loc[dt]; ok=x.notna()&z.notna()
      if ok.sum()>=8: rs.append(spearmanr(x[ok],z[ok]).statistic)
    q=pd.Series(rs).dropna(); print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan,'hit',(q>0).mean(),'n',len(q))
# subperiod stability
for start in ['2026-07-16','2027-01-01','2028-01-01','2029-01-01','2029-07-01']:
 q=a.loc[start:,'ic']; print(start,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
# artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20291213_low_downside_vol_signal.csv',index=False)
