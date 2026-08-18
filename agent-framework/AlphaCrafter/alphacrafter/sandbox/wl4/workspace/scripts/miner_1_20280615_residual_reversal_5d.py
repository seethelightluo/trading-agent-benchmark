import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=2400)
    if x is not None and len(x):
        z=x[['date','close']].copy(); z['date']=pd.to_datetime(z.date); z=z.drop_duplicates('date').set_index('date').close
        D[s]=z
p=pd.DataFrame(D).sort_index().ffill()
r=np.log(p).diff()
# rolling beta to SPX, residual cumulative 5d return; beta variation makes this non-common
m=r['SPX']
beta=r.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0)
res=r.sub(beta.mul(m,axis=0),axis=0)
f=-res.rolling(5).sum()
# forward non-overlapping daily cross-sectional IC, using signal at t and return t+1
rows=[]; turns=[]; dates=[]
for i in range(60,len(p)-1):
    d=p.index[i]; a=f.iloc[i]; y=r.iloc[i+1]
    q=pd.concat([a,y],axis=1).dropna();
    if len(q)>=8:
        rows.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
        ranks=q.iloc[:,0].rank(pct=True)
        if i>60:
            prev=f.iloc[i-1].reindex(q.index).rank(pct=True)
            turns.append((ranks-prev).abs().mean())
        dates.append(d)
ics=pd.Series(rows,index=dates).dropna()
print('dates',len(ics),'avg_n',len(U),'min_n',len(U),'coverage',float(p.notna().mean().mean()))
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f recent250_IC %.6f recent250_ICIR %.6f'%(ics.mean(),ics.mean()/ics.std(ddof=1), (ics>0).mean(), np.mean(turns), ics.tail(250).mean(), ics.tail(250).mean()/ics.tail(250).std(ddof=1)))
for h in [1,5,10,20]:
    aa=[]; dd=[]
    for i in range(60,len(p)-h):
        q=pd.concat([f.iloc[i], r.iloc[i+1:i+h+1].sum()],axis=1).dropna()
        if len(q)>=8: aa.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
    ss=pd.Series(aa).dropna(); print('horizon',h,'IC %.6f ICIR %.6f n %d'%(ss.mean(),ss.mean()/ss.std(ddof=1),len(ss)))
# regime halves
for name,ss in [('early',ics.iloc[:len(ics)//2]),('late',ics.iloc[len(ics)//2:])]: print(name,'ICIR',ss.mean()/ss.std(ddof=1),'IC',ss.mean(),'n',len(ss))
