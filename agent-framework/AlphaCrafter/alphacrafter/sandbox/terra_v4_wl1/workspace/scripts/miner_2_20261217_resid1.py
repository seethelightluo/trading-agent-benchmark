import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
 P[s]=d[d.index<=END]
P=pd.DataFrame(P).sort_index(); R=P.pct_change(); med=R.median(axis=1)
# lagged one-day residual reversal, normalized by lagged 20d volatility
F=-(R.sub(med,axis=0)).div(R.rolling(20,min_periods=15).std().replace(0,np.nan)).shift(1)
Y=P.shift(-1).div(P)-1
rows=[]
for dt in P.index:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((dt,z.f.corr(z.y,method='spearman'),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=a.ic
print('dates',len(x),'avgN',round(a.n.mean(),3),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for lo,hi,n in [('2020','2022','2020-22'),('2023','2024','2023-24'),('2025','2026-12-17','2025-26')]:
 q=x[(x.index>=lo)&(x.index<=hi)];print(n,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
F.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_2_20261217_resid1_signal.csv',index=False)
