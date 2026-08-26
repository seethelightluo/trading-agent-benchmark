import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=2200) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=np.log(px).diff()
# Interpretable trend acceleration: recent 20d return relative to prior 40d return,
# cross-sectionally demeaned and risk-scaled; all inputs lagged one session.
r20=r.rolling(20).sum(); r40=r.shift(20).rolling(40).sum(); vol=r.rolling(20).std()
acc=r20-r40
f=acc.sub(acc.mean(axis=1),axis=0)/(vol+1e-12)
f=f.shift(1)
print('instruments',len(px.columns),'dates',len(px))
for h in [1,5,10,20]:
 fr=np.log(px.shift(-h)/px); vals=[]; ns=[]; dates=[]
 for dt in f.index:
  a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   q=a[ok].corr(b[ok],method='spearman')
   if pd.notna(q): vals.append(q);ns.append(ok.sum());dates.append(dt)
 z=pd.Series(vals,index=dates)
 print('H',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('H10 thirds',[round(x.mean(),6) for x in np.array_split(z if h==20 else z,3)])
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=f.reset_index().rename(columns={'index':'date'});out.to_csv('scripts/miner_2_20340123_trend_acceleration_signal.csv',index=False)
print('signal_range',out.date.min(),out.date.max())
