import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
b=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:'2028-08-14'].ffill()
R=P.pct_change(); bm=R.mean(axis=1); beta=R.rolling(60,min_periods=30).cov(bm).div(bm.rolling(60,min_periods=30).var(),axis=0)
y=P.shift(-10)/P-1
f=-(P.pct_change(3)-beta.mul(bm.rolling(3).sum(),axis=0))
# Only express signal after a positive trailing five-day benchmark regime.
f[bm.rolling(5).sum()<=0]=np.nan
rows=[]
from scipy.stats import spearmanr
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=np.array([x[1] for x in rows]); dates=[x[0] for x in rows]
# turnover among active signals, day-over-day normalized rank changes
rank=f.rank(axis=1,pct=True); turnover=rank.diff().abs().mean(axis=1).loc[dates].mean()
print('dates',len(a),'avgN',np.mean([pd.concat([f.loc[d],y.loc[d]],axis=1).dropna().shape[0] for d in dates]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',f.notna().mean().mean(),'turnover',turnover)
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028-08-14')]:
 q=a[[str(d.date())[:4]>=lo and str(d.date())[:4]<=hi[:4] for d in dates]]
 print(lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
f.to_csv('scripts/miner_3_20280815_conditional_beta_residual3_up_signal.csv')
print('artifact written')
