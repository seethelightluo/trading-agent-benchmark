import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index();r=P.pct_change()
# Trend persistence: 20-day return weighted toward recent observations, scaled by downside deviation.
w=np.arange(1,21,dtype=float); w/=w.sum()
trend=r.rolling(20,min_periods=20).apply(lambda x: np.dot(x,w),raw=True)
down=r.where(r<0).rolling(40,min_periods=30).std()*np.sqrt(40)
sig=(trend/(down+1e-12)).shift(1).rank(axis=1,pct=True).sub(.5)
def test(h):
 y=P.shift(-h)/P-1; vals=[];ns=[];ds=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: vals.append(sig.loc[dt,ok].corr(y.loc[dt,ok],method='spearman'));ns.append(int(ok.sum()));ds.append(dt)
 return pd.Series(vals,index=pd.to_datetime(ds)),ns
for h in [1,5,10,20]:
 a,n=test(h);print('h',h,'dates',len(a),'avg_n %.2f'%np.mean(n),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
a,n=test(1);print('rows',len(P),'assets',len(P.columns),'coverage %.5f turnover %.5f'%((sig.notna()).mean().mean(),sig.diff().abs().mean().mean()))
for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]:print('regime',i,j,'IC %.8f ICIR %.8f'%(a.iloc[i:j].mean(),a.iloc[i:j].mean()/a.iloc[i:j].std(ddof=1)))
pd.DataFrame({'date':a.index,'ic':a.values,'n':n}).to_csv('scripts/miner_1_20310728_downside_trend_ic.csv',index=False)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20310728_downside_trend_signal.csv',index=False)
