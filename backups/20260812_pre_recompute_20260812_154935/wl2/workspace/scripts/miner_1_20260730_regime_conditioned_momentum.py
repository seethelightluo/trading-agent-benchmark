import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in symbols:
    d=get_stock_daily_data(s, days=1800)
    if d is not None and len(d)>30:
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); x=x.drop_duplicates('date').set_index('date').close
        frames[s]=x
p=pd.DataFrame(frames).sort_index()
# Candidate: 5d momentum whose sign is conditioned on aggregate 5d market regime;
# in negative breadth/regime, favor short-term mean reversion (contrarian), otherwise continuation.
r5=p.pct_change(5)
market=r5.mean(axis=1,skipna=True)
# smooth regime with cross-sectional breadth, requiring at least 8 assets
breadth=(r5>0).sum(axis=1)
signal=r5.where(breadth>=8, np.nan).mul(np.where(market>=0,1.,-1.),axis=0)
fwd=p.shift(-1)/p-1
rows=[]
for dt in signal.index:
    z=signal.loc[dt]; y=fwd.loc[dt]
    v=pd.concat([z,y],axis=1).dropna();
    if len(v)>=8 and v.iloc[:,0].nunique()>1 and v.iloc[:,1].nunique()>1:
        rows.append((dt,v.iloc[:,0].corr(v.iloc[:,1]),len(v), market.loc[dt], breadth.loc[dt]))
r=pd.DataFrame(rows,columns=['date','ic','n','regime','breadth']).set_index('date')
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover_proxy %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean(), signal.rank(axis=1,pct=True).diff().abs().stack().mean()))
for name,sub in [('2020-22',r.loc[:'2022-12-31']),('2023-24',r.loc['2023-01-01':'2024-12-31']),('2025+',r.loc['2025-01-01':]),('recent252',r.tail(252))]:
 print(name,len(sub), 'IC %.6f ICIR %.6f hit %.3f'%(sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1), (sub.ic>0).mean()))
for h in [1,5,10]:
 yy=p.shift(-h)/p-1; rr=[]
 for dt in signal.index:
  v=pd.concat([signal.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(v)>=8 and v.iloc[:,0].nunique()>1 and v.iloc[:,1].nunique()>1: rr.append(v.iloc[:,0].corr(v.iloc[:,1]))
 print('horizon',h,'n',len(rr),'IC %.6f ICIR %.6f'%(np.mean(rr),np.mean(rr)/np.std(rr,ddof=1)))
print('assets',len(frames), 'range',p.index.min(),p.index.max())
