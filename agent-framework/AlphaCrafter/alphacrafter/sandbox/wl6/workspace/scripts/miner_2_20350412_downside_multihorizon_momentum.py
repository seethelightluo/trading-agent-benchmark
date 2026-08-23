import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
    try:d=get_stock_daily_data(s,days=6000)
    except Exception:
        try:d=get_index_daily_data(s,days=6000)
        except Exception:d=None
    if d is not None and len(d)>180:
        d=d.copy();d.date=pd.to_datetime(d.date);F[s]=d.sort_values('date').set_index('date')
px=pd.DataFrame({s:d.close for s,d in F.items()}).sort_index(); lr=np.log(px).diff()
def sig(w):
    mom=np.log(px/px.shift(w)); dn=lr.where(lr<0,0).rolling(w,min_periods=max(20,w//2)).std()*np.sqrt(252)
    return (mom/dn.replace(0,np.nan)).shift(1)
# blended horizons, equal risk-normalized components
f=(sig(60).rank(axis=1,pct=True)+sig(120).rank(axis=1,pct=True))/2
out={}
for h in [5,10,20,40]:
 fw=np.log(px.shift(-h)/px); vals=[]; ds=[]; ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fw.loc[dt].notna()
  if ok.sum()>=8:
   c=f.loc[dt,ok].corr(fw.loc[dt,ok],method='spearman')
   if pd.notna(c): vals.append(c);ds.append(dt);ns.append(ok.sum())
 x=pd.Series(vals,index=pd.to_datetime(ds));out[h]=x
 print('horizon',h,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(len(x)),'hit',(x>0).mean())
print('data',px.index.min(),px.index.max(),'assets',len(F),'coverage',np.mean(ns)/15,'avg_names',np.mean(ns))
r=f.rank(axis=1,pct=True);print('turnover',r.diff().abs().mean(axis=1).mean())
for a,b in [('2020-01-01','2027-12-31'),('2028-01-01','2031-12-31'),('2032-01-01','2035-04-11')]:
 x=out[10][(out[10].index>=a)&(out[10].index<=b)];print('regime',a,b,'IC',x.mean(),'dates',len(x))
f.to_csv('scripts/miner_2_20350412_downside_multihorizon_momentum_signal.csv')
