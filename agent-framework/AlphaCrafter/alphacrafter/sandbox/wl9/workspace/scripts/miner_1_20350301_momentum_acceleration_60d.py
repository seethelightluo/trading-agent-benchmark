import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=6000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None and len(d)>200}).sort_index().ffill()
# acceleration: recent 20-session return minus preceding 40-session return, lagged one day
r20=px.pct_change(20); r60=px.pct_change(60)
f=(r20-r60/3).shift(1)
rows=[]
for i in range(1,len(px)-60):
 d=px.index[i]; nxt=px.iloc[i+1:i+61].iloc[-1]/px.iloc[i]/1-1
 # corrected forward from decision close to 60 sessions ahead
 nxt=px.iloc[i+60]/px.iloc[i]-1
 x=f.iloc[i]
 ok=x.notna()&nxt.notna()
 if ok.sum()>=8:
  rows.append((d,float(x[ok].corr(nxt[ok])),int(ok.sum())))
z=pd.DataFrame(rows,columns=['date','ic','n']); z['date']=pd.to_datetime(z.date)
for h in [10,20,40,60]:
 rr=[]
 for i in range(1,len(px)-h):
  x=f.iloc[i]; y=px.iloc[i+h]/px.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: rr.append(float(x[ok].corr(y[ok])))
 a=np.array(rr); print('H',h,'dates',len(a),'Nmean',round(float(z.n.mean()),2),'IC',round(float(np.nanmean(a)),6),'ICIR',round(float(np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(len(a))),6),'hit',round(float((a>0).mean()),4))
# coverage and rank turnover
valid=f.notna().sum(axis=1)/len(U); ranks=f.rank(pct=True,axis=1); tv=(ranks.diff().abs().mean(axis=1)).dropna()
print('coverage',round(float(valid.mean()),4),'turnover',round(float(tv.mean()),4),'instruments',len(px.columns),'date_range',px.index.min(),px.index.max())
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=z[(z.date>=a)&(z.date<=b)].ic.dropna(); print('REG',a,b,'dates',len(q),'IC',round(float(q.mean()),6),'ICIR',round(float(q.mean()/q.std(ddof=1)*np.sqrt(len(q))),6))
# save signal artifact
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20350301_momentum_acceleration_60d_signal.csv')
