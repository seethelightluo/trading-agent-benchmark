import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date); raw[s]=d.set_index('date').close
P=pd.DataFrame(raw).sort_index(); r=P.pct_change()
# Trend quality: medium-horizon return, penalized by downside volatility and rewarded
# by directional persistence; all components are completed-day and then lagged.
down=r.where(r<0,0).rolling(30,min_periods=15).std()
ret20=P.pct_change(20)
persist=(r.gt(0).rolling(20,min_periods=15).mean()-0.5)*2
sig=(ret20/(down*np.sqrt(30)+1e-8)*persist).shift(1)
# cross-sectional ranks make scale comparable across asset classes
sig=sig.rank(axis=1,pct=True)
y=P.shift(-1)/P-1
vals=[]; rows=[]
for dt in sig.index:
 v=sig.loc[dt].notna()&y.loc[dt].notna()
 if v.sum()>=8:
  c=sig.loc[dt,v].corr(y.loc[dt,v],method='spearman'); vals.append(c); rows.append((dt,c,int(v.sum())))
a=pd.Series(vals,index=[x[0] for x in rows]); print('dates',len(a),'assets',len(P.columns),'avg_n %.2f'%np.mean([x[2] for x in rows]))
print('IC %.8f ICIR %.8f hit %.5f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
for h in [5,10,20]:
 yy=P.shift(-h)/P-1; q=[]
 for dt in sig.index:
  v=sig.loc[dt].notna()&yy.loc[dt].notna()
  if v.sum()>=8:q.append(sig.loc[dt,v].corr(yy.loc[dt,v],method='spearman'))
 q=pd.Series(q); print('h',h,'dates',len(q),'IC %.8f ICIR %.8f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage %.5f turnover %.5f'%(sig.notna().mean().mean(),sig.diff().abs().mean().mean()))
n=len(a); print('regimes',*[round(a.iloc[i:j].mean(),8) for i,j in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
out='scripts/miner_1_20310616_trend_quality_persistence'; pd.DataFrame(rows,columns=['date','ic','n']).to_csv(out+'_ic.csv',index=False); sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv(out+'_signal.csv',index=False); print('signal_artifact='+out+'_signal.csv')
