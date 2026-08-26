import numpy as np, pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-11-03'); base=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
r=np.log(P).diff()
# Downside semideviation uses zero for non-negative returns, preserving coverage.
down=r.clip(upper=0).rolling(20,min_periods=15).std().replace(0,np.nan)
ret20=np.log(P/P.shift(20))
dd20=P/P.rolling(20,min_periods=15).max()-1
dd60=P/P.rolling(60,min_periods=40).max()-1
# Recovery-adjusted downside risk trend, lagged one day.
sig=((ret20/down)*(1+0.5*(dd20-dd60))).shift(1).replace([np.inf,-np.inf],np.nan)
for h in [5,10,20,40]:
 y=np.log(P.shift(-h)/P); rows=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 icir=q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan
 print('h',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4),'IC',round(q.ic.mean(),6),'ICIR',round(icir,6),'hit',round((q.ic>0).mean(),4))
 for label,sub in [('2025_26',q.loc['2025':'2026']),('2027_28',q.loc['2027':'2028']),('recent',q.loc['2028-10-01':])]:
  if len(sub)>1: print(label,len(sub),round(sub.ic.mean(),6),round(sub.ic.mean()/sub.ic.std(ddof=1),6))
rank=sig.rank(axis=1,pct=True); turnover=(rank.diff().abs().sum(axis=1)/2).dropna()
print('dates',len(P),'overall_coverage',round(sig.notna().sum(axis=1).mean()/15,4),'rank_turnover',round(turnover.mean(),6))
sig.stack().rename('signal').to_csv('scripts/miner_1_20291105_asymmetric_recovery_signal.csv')
y=np.log(P.shift(-10)/P); rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20291105_asymmetric_recovery_ic.csv',index=False)
