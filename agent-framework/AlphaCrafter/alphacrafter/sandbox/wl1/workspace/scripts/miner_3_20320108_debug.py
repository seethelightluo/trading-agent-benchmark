import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for s in U:
 d=get_stock_daily_data(s,days=5000); d['date']=pd.to_datetime(d.date); raw[s]=d.set_index('date').close
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff(); v=r.rolling(30).std(); r5=np.log(px/px.shift(5)); med=r5.median(axis=1); disp=r5.sub(med,axis=0).abs().median(axis=1); th=disp.rolling(120,min_periods=60).median(); active=(disp>th).astype(float); f=(-(r5.sub(med,axis=0))/(v*np.sqrt(5)+1e-12)*active).shift(1); y=np.log(px.shift(-10)/px)
print('valid r5',r5.notna().sum(axis=1).describe().to_dict()); print('disp',disp.notna().sum(),'active',active.sum(),'f valid',f.notna().sum().sum(),'y valid',y.notna().sum().sum()); print(px.tail())
for dt in f.index[-10:]:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna(); print(dt,len(z),z.iloc[:,0].std() if len(z) else 0)
