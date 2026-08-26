import numpy as np,pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2029-08-13'); base=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]; R=P.pct_change(); lag=R.shift(1)
# Short-term reversal, normalized by downside risk, with cross-sectional centering
ret10=P.shift(1)/P.shift(11)-1; down=lag.clip(upper=0).pow(2).rolling(40,min_periods=20).mean().pow(.5)
sig=-(ret10-ret10.median(axis=1).values[:,None])/down.replace(0,np.nan)
rows=[]
for dt in P.index:
 z=pd.concat([sig.loc[dt].rename('x'),(P.shift(-10)/P-1).loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.x.nunique()>1: rows.append((dt,len(z),spearmanr(z.x,z.y).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date').dropna(); print('dates',len(r),'avg_n',round(r.n.mean(),2),'coverage',round(r.n.mean()/15,4),'IC',round(r.ic.mean(),6),'ICIR',round(r.ic.mean()/r.ic.std(ddof=1),6),'hit',round((r.ic>0).mean(),4))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-08-13')]:
 q=r.loc[a:b].ic; print('regime',a,b,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6)); sig.to_csv('scripts/miner_1_20290827_downside_reversal_signal.csv'); r.to_csv('scripts/miner_1_20290827_downside_reversal_ic.csv')
