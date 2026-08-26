import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0: d=get_index_daily_data(s,4000)
 return d
px={}
for s in U:
 d=fetch(s)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
D=pd.concat(px,axis=1).sort_index().ffill()
# downside-risk-adjusted 60d momentum: return divided by downside semideviation over trailing 60 daily returns
rets=D.pct_change()
rollret=D/D.shift(60)-1
neg=rets.where(rets<0,0.0)
down=np.sqrt((neg.pow(2).rolling(60,min_periods=40).mean()))*np.sqrt(60)
sig=rollret/down
# lag signal one completed day explicitly
sig=sig.shift(1)
# forward 10 trading rows
fwd=D.shift(-10)/D-1
rows=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]
 z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# current-date constrained automatically by data endpoint; report recent and regimes
for label,a,b in [('all','2020-01-01','2029-07-02'),('2025-01-01','2025-01-01','2026-12-31'),('2027-01-01','2027-01-01','2028-08-31'),('recent','2028-09-01','2029-07-02')]:
 q=r.loc[a:b,'ic'].dropna(); print(label,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan,(q>0).mean())
print('dates',len(r),'avg_n',r.n.mean(),'coverage',sig.notna().stack().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().stack().mean())
for h in [1,5,10,20]:
 yy=D.shift(-h)/D-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,np.nanmean(rr),len(rr))
# artifact
out=sig.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_2_20290702_downside_momentum60_signal.csv')
