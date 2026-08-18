import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float) for s in U}
p=pd.DataFrame(D).sort_index().ffill().loc[:'2034-06-23']; r=p.pct_change()
# Breadth-confirmed trend: 20d return, gated by cross-asset directional breadth over 40d.
# When breadth confirms direction, retain trend; otherwise attenuate. Lagged one day.
t=p.pct_change(20); breadth=(t>0).mean(axis=1); gate=(2*breadth-1)
sig=(t.mul(gate,axis=0)).shift(1)
fwd=p.shift(-10)/p-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'turnover',sig.rank(pct=True).diff().abs().mean().mean())
for h in [5,10,20,40]:
 fw=p.shift(-h)/p-1;a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=pd.Series(a);print('decay',h,a.mean(),a.mean()/a.std(ddof=1),len(a))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
 x=q.loc[a:b,'ic'];print('regime',a,b,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>2 else np.nan)
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20340623_breadth_trend_signal.csv',index=False)
