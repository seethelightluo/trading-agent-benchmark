import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
neg=r.clip(upper=0)
down=np.sqrt((neg.pow(2)).rolling(40,min_periods=30).mean())*np.sqrt(40)
trend=r.rolling(20,min_periods=20).sum()
sig=(trend/(down+1e-12)).shift(1).rank(axis=1,pct=True).sub(.5)
print('rows',len(P),'assets',len(P.columns),'date_start',P.index.min(),'date_end',P.index.max())
results={}
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; vals=[]; ns=[]; dates=[]
 for dt in sig.index:
  vv=sig.loc[dt].notna()&y.loc[dt].notna()
  if vv.sum()>=8:
   vals.append(sig.loc[dt,vv].corr(y.loc[dt,vv],method='spearman')); ns.append(int(vv.sum())); dates.append(dt)
 a=pd.Series(vals,index=pd.to_datetime(dates)); results[h]=(a,ns)
 print('h',h,'dates',len(a),'avg_n %.2f'%np.mean(ns),'IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
 if h in (1,5): pd.DataFrame({'date':a.index,'ic':a.values,'n':ns}).to_csv(f'scripts/miner_1_20310811_downside_trend_ic_{h}d.csv',index=False)
a,ns=results[1]
for i,j in [(0,len(a)//3),(len(a)//3,2*len(a)//3),(2*len(a)//3,len(a))]: print('regime',i,j,'IC %.8f ICIR %.8f'%(a.iloc[i:j].mean(),a.iloc[i:j].mean()/a.iloc[i:j].std(ddof=1)))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20310811_downside_trend_signal.csv',index=False)
print('coverage %.5f turnover %.5f'%((sig.notna()).mean().mean(),sig.diff().abs().mean().mean()))
