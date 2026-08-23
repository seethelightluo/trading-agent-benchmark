import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict()['watch_list']
# VIX observation file, safely lagged and aligned
v=pd.read_csv('../persistent/index_data/VIX.csv')
v['date']=pd.to_datetime(v['date']); v=v.set_index('date')['close'].sort_index()
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None: continue
 d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
 x=d['close'].pct_change(); vr=v.reindex(d.index).ffill().pct_change()
 # lagged 60d covariance beta to VIX, with shock intensity; lower beta is defensive
 beta=x.rolling(60,min_periods=40).cov(vr)/vr.rolling(60,min_periods=40).var()
 shock=(vr.rolling(20,min_periods=15).mean()-vr.rolling(120,min_periods=60).mean())/vr.rolling(120,min_periods=60).std()
 fac=-(beta.shift(1))*(1+shock.clip(lower=0).shift(1))
 fwd=x.shift(-1)
 z=pd.DataFrame({'f':fac,'r':fwd,'s':s}).dropna(); rows.append(z)
z=pd.concat(rows)
ics=z.groupby(level=0).apply(lambda q: q['f'].corr(q['r']) if len(q)>=8 else np.nan).dropna()
# date index survives from each frame
print('dates',len(ics),'avg_names',z.groupby(level=0).size().mean(),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean(),'coverage',len(z)/(len(ics)*len(U)))
print('regimes',[(str(a),round(b.mean(),4),round(b.mean()/b.std(ddof=1),3),len(b)) for a,b in ics.groupby(ics.index.year)])
print('turnover proxy',z.groupby('s')['f'].apply(lambda x:(x.diff().abs()>x.rolling(20).std()).mean()).mean())
