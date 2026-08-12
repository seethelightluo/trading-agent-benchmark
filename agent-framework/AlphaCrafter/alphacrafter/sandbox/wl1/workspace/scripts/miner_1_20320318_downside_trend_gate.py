import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,days=1800)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=1800)
 if d is not None: rows.append(d[['date','close']].assign(symbol=s))
p=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=np.log(p).diff()
# Downside-risk-adjusted medium momentum, with a slow positive-trend quality gate; all lagged.
ret20=p.pct_change(20); ret60=p.pct_change(60); dn=r.where(r<0).rolling(40).std(); tot=r.rolling(40).std()
# retain sign and reward momentum when slow trend agrees; fallback risk avoids missing cross-section
risk=dn.fillna(tot).replace(0,np.nan)
f=(ret20/risk)*np.where(ret60>0,1.0,-0.35); f=f.shift(1)
ics=[]; dates=[]; ns=[]
for dt in p.index:
 y=p.shift(-10).loc[dt]/p.loc[dt]-1 if dt in p.index else None
 if y is None: continue
 ok=f.loc[dt].notna()&y.notna()
 if ok.sum()>=8:
  v=f.loc[dt][ok].corr(y[ok])
  if np.isfinite(v): ics.append(v); dates.append(dt); ns.append(ok.sum())
ic=np.array(ics)
print('candidate=downside_risk_adjusted_trend_gate_20d; dates=%d avg_n=%.3f coverage=%.4f'%(len(ic),np.mean(ns),np.mean(ns)/15))
print('IC=%.6f ICIR=%.6f hit=%.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1),np.mean(ic>0)))
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; aa=[]
 for dt in p.index:
  ok=f.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:
   v=f.loc[dt][ok].corr(yy.loc[dt][ok]);
   if np.isfinite(v): aa.append(v)
 aa=np.array(aa); print('decay',h,'%.6f'%aa.mean(),'n',len(aa))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2025-12-31'),('2026-01-01','2028-12-31'),('2029-01-01','2030-12-31'),('2031-01-01','2032-03-18')]:
 vals=[]
 for dt in p.loc[a:b].index:
  y=p.shift(-10).loc[dt]/p.loc[dt]-1; ok=f.loc[dt].notna()&y.notna()
  if ok.sum()>=8:
   v=f.loc[dt][ok].corr(y[ok]);
   if np.isfinite(v): vals.append(v)
 vals=np.array(vals); print('regime',a[:4],len(vals),'IC %.6f ICIR %.6f'%(vals.mean(),vals.mean()/vals.std(ddof=1)) if len(vals)>1 else 'NA')
# rank turnover at dates
turn=[]; prev=None
for dt in dates:
 q=f.loc[dt].rank(pct=True)
 if prev is not None: turn.append((q-prev).abs().mean())
 prev=q
print('turnover=%.6f'%np.nanmean(turn))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20320318_downside_trend_gate_signal.csv',index=False)
