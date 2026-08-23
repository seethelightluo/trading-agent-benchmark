import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict()['watch_list']
def macro(name):
 q=pd.read_csv('../persistent/index_data/'+name+'.csv'); q['date']=pd.to_datetime(q['date']); return q.set_index('date')['close'].sort_index()
v=macro('VIX'); dxy=macro('DXY'); rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None: continue
 d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index(); r=d.close.pct_change()
 vv=v.reindex(d.index).ffill(); dx=dxy.reindex(d.index).ffill()
 vz=(vv-vv.rolling(120,min_periods=60).mean())/vv.rolling(120,min_periods=60).std()
 dz=(dx-dx.rolling(120,min_periods=60).mean())/dx.rolling(120,min_periods=60).std()
 # reversal favored only in calm/weak-dollar regime; defensive opposite under stress
 f=(-r.rolling(5).sum()/r.rolling(20).std())*(1-0.35*vz.clip(lower=0)-0.15*dz.clip(lower=0))
 rows.append(pd.DataFrame({'f':f.shift(1),'r':r.shift(-1),'s':s}).dropna())
z=pd.concat(rows); ic=z.groupby(level=0).apply(lambda q:q.f.corr(q.r) if len(q)>=8 else np.nan).dropna()
print('dates',len(ic),'avg_names',z.groupby(level=0).size().mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'coverage',len(z)/(len(ic)*len(U)))
print('regimes',[(int(y),round(x.mean(),4),round(x.mean()/x.std(ddof=1),3),len(x)) for y,x in ic.groupby(ic.index.year)])
print('turnover',z.groupby('s').f.apply(lambda x:(x.diff().abs()>x.rolling(20).std()).mean()).mean())
