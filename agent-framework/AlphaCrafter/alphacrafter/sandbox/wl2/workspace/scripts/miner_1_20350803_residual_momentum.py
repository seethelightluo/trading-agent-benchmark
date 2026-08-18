import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d)>=150:return d
  except Exception: pass
 return None
xs={s:fetch(s) for s in U}; print('lengths',{s:(len(d) if d is not None else 0) for s,d in xs.items()})
# candidate: beta-neutral residual momentum, 20d return less rolling beta*cross-sectional equal-weight return, volatility scaled
P={}
for s,d in xs.items():
 if d is not None:
  q=d.assign(date=pd.to_datetime(d.date)).sort_values('date').set_index('date').close.astype(float)
  P[s]=q.pct_change()
r=pd.DataFrame(P).sort_index(); m=r.mean(axis=1); rows=[]
for s in U:
 if s not in P: continue
 rs=r[s]; beta=rs.rolling(60).cov(m)/m.rolling(60).var(); resid=rs-beta*m
 f=resid.rolling(20).sum()/rs.rolling(20).std()
 fw=P[s].shift(-10).rolling(10).sum().shift(-9) # future 10 daily returns aligned at t
 # safer direct close forward
 p=xs[s].assign(date=pd.to_datetime(xs[s].date)).sort_values('date').set_index('date').close.astype(float)
 fw=p.shift(-10)/p-1
 for dt in f.index:
  if pd.notna(f.loc[dt]) and pd.notna(fw.loc[dt]): rows.append((dt,s,f.loc[dt],fw.loc[dt]))
z=pd.DataFrame(rows,columns=['date','s','factor','fwd']); ii=[]
for dt,g in z.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1: ii.append((dt,g.factor.corr(g.fwd),len(g)))
ii=pd.DataFrame(ii,columns=['date','ic','n']).set_index('date')
print('dates',len(ii),'avgN',ii.n.mean(),'coverage',len(z)/(len(ii)*15),'IC %.6f ICIR %.6f hit %.4f'%(ii.ic.mean(),ii.ic.mean()/ii.ic.std(),(ii.ic>0).mean()))
for a,b in [('2020','2025'),('2026','2030'),('2031','2035')]:
 q=ii.loc[a:b].ic;print(a,b,len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
print('decay')
for h in [1,3,5,10,20]:
 vv=[]
 for s in U:
  if s not in P:continue
  rs=r[s]; beta=rs.rolling(60).cov(m)/m.rolling(60).var(); resid=rs-beta*m; f=resid.rolling(20).sum()/rs.rolling(20).std()
  p=xs[s].assign(date=pd.to_datetime(xs[s].date)).sort_values('date').set_index('date').close.astype(float); fw=p.shift(-h)/p-1
  for dt in f.index:
   if pd.notna(f.loc[dt]) and pd.notna(fw.loc[dt]):vv.append((dt,s,f.loc[dt],fw.loc[dt]))
 q=pd.DataFrame(vv,columns=['date','s','f','fw']); v=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fw.nunique()>1:v.append(g.f.corr(g.fw))
 print(h,len(v),'IC %.6f ICIR %.6f'%(np.nanmean(v),np.nanmean(v)/np.nanstd(v)))
z.to_csv('../persistent/miner_1_20350803_residual_momentum_signal.csv',index=False)
