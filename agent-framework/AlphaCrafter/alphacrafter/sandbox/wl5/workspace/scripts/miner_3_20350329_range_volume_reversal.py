import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={};hi={};lo={};vol={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is not None and len(d)>=100:
  x=d.set_index('date'); cl[s]=x.close.astype(float); hi[s]=x.high.astype(float); lo[s]=x.low.astype(float); vol[s]=x.volume.astype(float)
P=pd.DataFrame(cl).sort_index(); H=pd.DataFrame(hi).reindex(P.index); L=pd.DataFrame(lo).reindex(P.index); V=pd.DataFrame(vol).reindex(P.index)
r=P.pct_change(); cs=r.sub(r.mean(axis=1),axis=0)
res=(-cs.rolling(10,min_periods=8).sum()/(cs.rolling(40,min_periods=25).std()*np.sqrt(10)+1e-12)).clip(-8,8)
prev=P.shift(1); tr=pd.DataFrame(np.maximum((H-L).values,np.maximum((H-prev).abs().values,(L-prev).abs().values)),index=P.index,columns=P.columns)
rz=np.log1p(tr/P.clip(lower=1e-12)); z=(rz-rz.rolling(40,min_periods=25).mean())/(rz.rolling(40,min_periods=25).std()+1e-12)
# Abnormal volume confirms shocks; bounded interaction remains interpretable and causal.
lv=np.log1p(V.replace(0,np.nan)); vz=(lv-lv.rolling(40,min_periods=25).mean())/(lv.rolling(40,min_periods=25).std()+1e-12)
sig=res*(1+0.25*np.tanh(z)*np.tanh(vz)).clip(.75,1.25)
print('assets',len(P.columns),'rows',len(P))
def ev(h):
 q=P.shift(-h)/P-1; a=[];ds=[];ns=[]
 for dt in sig.index:
  x=pd.concat([sig.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(x)>=8:
   c=x.iloc[:,0].corr(x.iloc[:,1],method='spearman')
   if np.isfinite(c): a.append(c);ds.append(dt);ns.append(len(x))
 return np.array(a),pd.DatetimeIndex(ds),np.array(ns)
for h in [5,10,20]:
 a,ds,ns=ev(h); print('horizon',h,'dates',len(a),'start',ds[0].date(),'end',ds[-1].date(),'mean_n',round(ns.mean(),3),'coverage',round(ns.mean()/15,6),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),6))
 if h==20:
  for x,y in [('2023-09-14','2025-12-31'),('2026-01-01','2030-12-31'),('2031-01-01','2035-03-01')]:
   w=a[(ds>=pd.Timestamp(x))&(ds<=pd.Timestamp(y))]; print('regime',x,y,'dates',len(w),'IC',round(w.mean(),6) if len(w) else None)
  ranks=pd.DataFrame([sig.loc[d].rank(pct=True) for d in ds],index=ds); print('turnover',round(ranks.diff().abs().mean().mean(),6))
  pd.DataFrame([(dt,s,float(sig.loc[dt,s])) for dt in sig.index for s in sig.columns if pd.notna(sig.loc[dt,s])],columns=['date','symbol','factor_value']).to_csv('scripts/miner_3_20350329_range_volume_reversal_signal.csv',index=False)
