import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return d.close
P=pd.concat({s:load(s) for s in U},axis=1,sort=True).loc[:'2026-07-15']; R=np.log(P).diff(); b=R.mean(axis=1); dn=b<0
F=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U:
 a=R[s]; down_count=dn.astype(float).rolling(60,min_periods=30).sum(); up_count=(~dn).astype(float).rolling(60,min_periods=30).sum()
 dm=a.where(dn).fillna(0).rolling(60,min_periods=30).sum()/down_count
 um=a.where(~dn).fillna(0).rolling(60,min_periods=30).sum()/up_count
 F[s]=dm-um
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); z=[]; ns=[]; dates=[]
 for dt in F.index:
  g=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: z.append(g.f.corr(g.y,method='spearman')); ns.append(len(g)); dates.append(dt)
 z=np.asarray(z); dates=pd.DatetimeIndex(dates); sd=np.std(z,ddof=1)
 print('h',h,'dates',len(z),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(z),np.mean(z)/sd,np.mean(z>0)))
 if h==1:
  print('coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
  for label,m in [('2020-22',dates<='2022-12-31'),('2023-24',(dates>='2023-01-01')&(dates<='2024-12-31')),('2025-26',dates>='2025-01-01')]:
   a=z[m];print(label,len(a),'IC %.6f ICIR %.6f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1)))
F.to_csv('scripts/miner_2_20260924_downside_resilience_signal.csv');print('artifact written')
