import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}; hi={}; lo={}
for s in U:
 d=get_stock_daily_data(s,4000); d['date']=pd.to_datetime(d.date); x=d.set_index('date'); cl[s]=x.close.astype(float); hi[s]=x.high.astype(float); lo[s]=x.low.astype(float)
p=pd.concat(cl,axis=1).sort_index(); H=pd.concat(hi,axis=1).reindex(p.index); L=pd.concat(lo,axis=1).reindex(p.index)
r=p.pct_change();
# Trend continuation when recent returns are efficient (few sign changes), scaled by downside risk.
ret5=r.rolling(5,min_periods=4).sum(); path=r.abs().rolling(10,min_periods=8).sum(); eff=ret5/(path+1e-12)
down=r.where(r<0,0).pow(2).rolling(20,min_periods=12).mean().pow(.5)
f=(eff/(down+1e-8)).shift(1); y=p.shift(-10)/p-1
ics=[]; cov=[]; turn=[]; prev=None; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');
  if np.isfinite(ic):
   ics.append(ic); cov.append(len(z)/15); dates.append(dt)
   q=f.loc[dt].rank(pct=True)
   if prev is not None: turn.append((q-prev).abs().mean())
   prev=q
A=np.array(ics); print('dates',len(A),'instruments',15,'avg_cov',np.mean(cov),'IC',A.mean(),'ICIR',A.mean()/A.std(ddof=1),'hit',(A>0).mean(),'turn',np.mean(turn))
for a in np.array_split(A,4):print('regime',len(a),a.mean(),a.mean()/a.std(ddof=1))
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=z.iloc[:,0].corr(z.iloc[:,1],method='spearman');
   if np.isfinite(v):q.append(v)
 q=np.array(q); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20291231_efficiency_downside_signal.csv',index=False)
