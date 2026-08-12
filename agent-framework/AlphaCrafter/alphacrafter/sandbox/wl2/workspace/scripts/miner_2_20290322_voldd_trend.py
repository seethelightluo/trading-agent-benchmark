import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            x=f(symbol=s,days=2600)
            if x is not None and len(x)>100: return x[['date','close']].copy()
        except Exception: pass
    return None
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
print('assets',len(D),{s:len(x) for s,x in D.items()})
px=pd.concat([x.set_index('date').close.rename(s) for s,x in D.items()],axis=1).sort_index().ffill()
r=np.log(px/px.shift(1))
# one interpretable idea: volatility-adjusted trend with drawdown penalty; all inputs lagged one bar
# score = 60d return / 20d vol, penalized by current drawdown from 120d high (less drawdown is better)
vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
trend=np.log(px/px.shift(60))
dd=px/px.rolling(120,min_periods=60).max()-1
fac=(trend/vol.replace(0,np.nan))*(1+dd.clip(-.5,0))
fac=fac.shift(1)
fr=px.shift(-1)/px-1
rows=[]
for d in fac.index:
    a=fac.loc[d]; b=fr.loc[d]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'avgN',x.n.mean(),'coverage',x.n.mean()/15)
for label,y in [('d1',x.ic),('h5',fac.shift(0).rolling(1).mean().index.to_series().map(lambda d: np.nan))]:
    if label=='d1':
      print(label,'IC %.6f ICIR %.6f hit %.3f'%(y.mean(),y.mean()/y.std(),(y>0).mean()))
for h in [3,5,10,20]:
    f=fac
    fw=px.shift(-h)/px-1
    rr=[]
    for d in f.index:
      z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
      if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    q=pd.Series(rr).dropna(); print('h%d IC %.6f ICIR %.6f'%(h,q.mean(),q.mean()/q.std()))
# turnover as rank topological signal change
ranks=fac.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).dropna().mean()
print('turnover',turn,'period',x.index.min(),x.index.max())
# regimes yearly
for yr,g in x.groupby(x.index.year): print('year',yr,'n',len(g),'IC',g.ic.mean(),'ICIR',g.ic.mean()/g.ic.std())
