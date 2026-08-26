import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    d=get_stock_daily_data(s,2400)
    if d is not None and len(d)>100:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index()
# candidate: residualized short reversal, residual to cross-sectional median 20d movement
R=P.pct_change()
# compute factor at date t: negative 5d return after removing common cross-sectional median 5d return
f=(-R.rolling(5).sum()).sub((-R.rolling(5).sum()).median(axis=1),axis=0)
# alternate intended signal: residual reversal scaled by 20d vol
vol=R.rolling(20).std()*np.sqrt(20)
f=f.div(vol.replace(0,np.nan))
fr=P.shift(-10).div(P)-1
rows=[]
for dt in P.index:
    x=f.loc[dt]; y=fr.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,float(z.iloc[:,0].corr(z.iloc[:,1])),len(z)))
D=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,sub in [('all',D),('2025-26',D.loc['2025':'2026']),('2027-28',D.loc['2027':'2028']),('recent',D.loc['2028-09-01':])]:
    if len(sub): print(label,'dates',len(sub),'avg_n',round(sub.n.mean(),2),'IC',round(sub.ic.mean(),5),'ICIR',round(sub.ic.mean()/sub.ic.std(ddof=1),5),'hit',round((sub.ic>0).mean(),4))
# turnover: average rank signal changes across adjacent valid dates
rank=f.rank(axis=1,pct=True)
to=(rank.diff().abs().mean(axis=1)/2).dropna()
print('coverage',round(D.n.mean()/15,5),'turnover',round(to.reindex(D.index).mean(),5),'first',D.index.min(),'last',D.index.max())
# decay 1,5,10,20 forward returns
for h in [1,5,10,20]:
    yy=P.shift(-h).div(P)-1; vals=[]
    for dt in P.index:
      z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
      if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
    a=np.array(vals); print('decay',h,'dates',len(a),'IC',round(np.nanmean(a),5),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),5))
