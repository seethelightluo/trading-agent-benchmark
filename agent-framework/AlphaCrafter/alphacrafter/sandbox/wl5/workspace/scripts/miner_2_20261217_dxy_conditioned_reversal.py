import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Macro-conditioned reversal: fade prior close return when DXY direction supports broad risk reversal.
try:
 m=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 dxy=m['close'].pct_change()
except Exception:
 m=None
A=[]; total=0
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.drop_duplicates('date').set_index('date').sort_index();total+=len(d)
 r=d.close.pct_change()
 if m is None: continue
 z=pd.DataFrame({'r':r,'dx':dxy}).dropna()
 # signal formed after t close, use contemporaneous completed DXY move; negative asset return, signed by DXY move
 sig=-(z.r)*np.sign(-z.dx)
 fr=d.close.shift(-1)/d.close-1
 q=pd.DataFrame({'date':z.index,'sig':sig,'fr':fr.reindex(z.index)}).dropna().assign(s=s)
 A.append(q)
x=pd.concat(A,ignore_index=True); out=[];ns=[]; dates=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.sig.nunique()>1 and g.fr.nunique()>1:
  v=g.sig.corr(g.fr,method='spearman')
  if np.isfinite(v):out.append(v);ns.append(len(g));dates.append(dt)
a=np.array(out);print('dates',len(a),'avg_n',round(np.mean(ns),2),'assets',x.s.nunique(),'coverage_rows',round(len(x)/total,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=a[[lo<=d.year<=hi for d in dates]];print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
rank=x.pivot(index='date',columns='s',values='sig').rank(axis=1,pct=True);print('turnover',round(rank.diff().abs().mean(axis=1).mean(),4))
x.to_csv('scripts/miner_2_20261217_dxy_conditioned_reversal_signal.csv',index=False)
