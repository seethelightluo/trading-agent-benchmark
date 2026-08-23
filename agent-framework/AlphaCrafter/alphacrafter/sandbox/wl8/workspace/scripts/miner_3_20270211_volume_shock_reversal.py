import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# high-volume shock reversal: recent 3d return reversed and scaled by volume surprise
D={}
for s in U:
    x=get_stock_daily_data(s,days=2700)
    if x is None or len(x)<100: continue
    x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index()
    c=pd.to_numeric(x['close'],errors='coerce'); v=pd.to_numeric(x['volume'],errors='coerce')
    r=c.pct_change(); vs=(v/(v.rolling(20,min_periods=10).median())).replace([np.inf,-np.inf],np.nan)
    # robust signed shock: reversal of 3d move, enhanced by unusual volume, clipped
    f=-(c.pct_change(3))*np.log(vs.clip(lower=.25,upper=4.0))
    D[s]=pd.DataFrame({'f':f,'r1':c.pct_change().shift(-1),'r5':c.pct_change(5).shift(-5),'r10':c.pct_change(10).shift(-10)})
all_dates=sorted(set().union(*[set(x.index) for x in D.values()]))
rows=[]
for dt in all_dates:
    z=[]
    for s,x in D.items():
        if dt in x.index and np.isfinite(x.loc[dt,'f']) and np.isfinite(x.loc[dt,'r1']): z.append((s,x.loc[dt]))
    if len(z)>=8:
        for s,q in z: rows.append({'date':dt,'s':s,**q.to_dict()})
a=pd.DataFrame(rows); print('dates',a.date.nunique(),'rows',len(a),'avg_names',a.groupby('date').size().mean())
def calc(y):
    ic=a.groupby('date').apply(lambda g:g.f.corr(g[y])).dropna()
    return len(ic),ic.mean(),ic.mean()/ic.std(ddof=1), (ic>0).mean()
for y in ['r1','r5','r10']: print(y,calc(y))
# coverage and rank turnover
print('coverage',len(a)/max(1,a.date.nunique()*15))
r=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True)
print('rank_turnover',r.diff().abs().mean().mean())
for yr,g in a.groupby(a.date.dt.year):
    ic=g.groupby('date').apply(lambda x:x.f.corr(x.r1)).dropna(); print('regime',yr,len(ic),round(ic.mean(),5),round(ic.mean()/ic.std(ddof=1),4))
