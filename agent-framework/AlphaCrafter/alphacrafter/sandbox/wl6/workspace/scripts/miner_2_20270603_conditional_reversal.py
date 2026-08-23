import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index()
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std(); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
# conditional short reversal: negative 5d return/vol, activated when lagged VIX is above its 60d median
base=(-(P/P.shift(5)-1)/(vol*np.sqrt(252))).shift(1)
reg=(vix.shift(1)>vix.shift(1).rolling(60,min_periods=30).median()).astype(float)
F=base.mul(reg,axis=0); R=P.shift(-10)/P-1
for label,X in [('conditional_vix_reversal',F),('plain_10d_volscaled_reversal',(-(P/P.shift(10)-1)/(vol*np.sqrt(252))).shift(1))]:
 vals=[]; ns=[]; cov=[]; turns=[]; prev=None
 for dt in X.index:
  x,y=X.loc[dt],R.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   z=spearmanr(x[ok],y[ok]).statistic; vals.append(z);ns.append(ok.sum());cov.append(ok.mean()); q=x.rank(pct=True); turns.append(np.mean(np.abs(q-(prev if prev is not None else q))));prev=q
 a=np.array(vals); print(label,'dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(cov),'IC',a.mean(),'ICIR',a.mean()/(a.std(ddof=1)/np.sqrt(len(a))),'hit',np.mean(a>0),'turnover',np.mean(turns))
 for n,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027','2027-01-01','2027-06-02')]:
  q=[spearmanr(X.loc[d][(X.loc[d].notna()&R.loc[d].notna())],R.loc[d][(X.loc[d].notna()&R.loc[d].notna())]).statistic for d in X.index[(X.index>=lo)&(X.index<=hi)] if (X.loc[d].notna()&R.loc[d].notna()).sum()>=8]
  q=np.array(q); print(' ',n,len(q),q.mean() if len(q) else np.nan,(q.mean()/(q.std(ddof=1)/np.sqrt(len(q)))) if len(q)>1 else np.nan)
 if label.startswith('conditional'):
  X.reset_index().to_csv('scripts/miner_2_20270603_conditional_reversal_signal.csv',index=False)
