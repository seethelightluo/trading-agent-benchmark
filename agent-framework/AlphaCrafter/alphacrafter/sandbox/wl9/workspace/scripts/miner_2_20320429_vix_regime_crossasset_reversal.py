import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); px[s]=x.set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date)
vc=[c for c in v.columns if c.lower() in ('close','adj_close','value')]
if not vc: vc=[c for c in v.columns if c!='date']
vix=pd.to_numeric(v.set_index('date')[vc[0]],errors='coerce').reindex(p.index).ffill(); vp=vix.rolling(252,min_periods=100).rank(pct=True)
rr=p.pct_change(20); base=-(rr.sub(rr.median(axis=1),axis=0)); sig=base.div(r.rolling(60,min_periods=30).std()); sig=sig.mul(0.75+vp.shift(1),axis=0).shift(1)
for h in [5,10,20,40,60]:
 f=p.shift(-h)/p-1; ics=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 a=np.array(ics); a=a[np.isfinite(a)]; print(f'h={h} dates={len(a)} avgN={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/a.std(ddof=1):.6f} hit={(a>0).mean():.4f}')
valid=sig.notna().sum(axis=1)>=8; print('range',sig.index.min().date(),sig.index.max().date(),'dates',valid.sum(),'avg_names',sig.loc[valid].notna().sum(axis=1).mean(),'coverage',sig.stack().notna().mean())
print('turnover_proxy',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f=p.shift(-20)/p-1
for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2030-12-31'),('2031-01-01','2032-04-28')]:
 ix=[]
 for dt in sig.loc[a:b].index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: ix.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 ix=np.array(ix); ix=ix[np.isfinite(ix)]; print('regime',a,b,'dates',len(ix),'IC',ix.mean(),'ICIR',ix.mean()/ix.std(ddof=1))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320429_vix_regime_crossasset_reversal_signal.csv',index=False)
