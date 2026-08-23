import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,days=3300)
 if d is not None and len(d):
  q=d[['date','close']].copy(); q.date=pd.to_datetime(q.date); frames[s]=q.drop_duplicates('date').set_index('date').close.astype(float).sort_index()
p=pd.concat(frames,axis=1).sort_index(); r=p.pct_change()
# Short-horizon reversal damped by recent volatility: fade 20d return / 40d total volatility.
f=-(p/p.shift(20)-1)/(r.rolling(40,min_periods=25).std()*np.sqrt(252)+1e-8)
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0)
fr=p.shift(-10)/p-1; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); rank=f.rank(axis=1,pct=True)
print('candidate=vol_adjusted_reversal_20d'); print('dates',len(a),'mean_instruments',a.n.mean(),'coverage_pct',a.n.mean()/15*100)
print('IC %.8f ICIR %.8f hit %.6f turnover %.8f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean(),rank.diff().abs().mean(axis=1).dropna().mean()))
for name,sub in [('2020-2024',a.loc[:'2024-12-31']),('2025-2027',a.loc['2025':'2027-12-31']),('2028-2029',a.loc['2028':'2029-12-31']),('2030-2032',a.loc['2030':'2032-06-09'])]:
 print(name,'n',len(sub),'IC',sub.ic.mean(),'ICIR',sub.ic.mean()/sub.ic.std(ddof=1) if len(sub)>1 else np.nan,'hit',(sub.ic>0).mean())
f.loc[:'2032-06-09'].rename_axis('date').to_csv('scripts/miner_2_20320610_vol_adjusted_reversal_20d_signal.csv')
