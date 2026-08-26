import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is None or len(d)<300: continue
    d=d.sort_values('date').drop_duplicates('date'); d['r']=np.log(d.close.astype(float)).diff()
    # lag-safe factor: acceleration, recent 10d trend vs slow 60d trend, both ending 5 sessions ago
    r10=np.log(d.close/d.close.shift(5)).shift(5)
    r60=np.log(d.close/d.close.shift(5+55)).shift(5)
    v40=d.r.shift(1).rolling(40,min_periods=25).std()
    f=(r10-r60/6)/v40
    # preserve dates and forward returns
    for i in range(len(d)):
        if pd.notna(f.iloc[i]):
            rows.append((d.date.iloc[i],s,float(f.iloc[i]),float(np.log(d.close.iloc[min(i+1,len(d)-1)]/d.close.iloc[i])) if i+1<len(d) else np.nan, *[float(np.log(d.close.iloc[min(i+h,len(d)-1)]/d.close.iloc[i])) if i+h<len(d) else np.nan for h in [5,10,20]]))
x=pd.DataFrame(rows,columns=['date','symbol','factor','f1','f5','f10','f20'])
print('rows',len(x),'dates',x.date.nunique(),'instruments',x.symbol.nunique())
for h in ['f1','f5','f10','f20']:
  vals=[]
  for dt,g in x.groupby('date'):
    g=g.dropna(subset=['factor',h])
    if len(g)>=8: vals.append(g.factor.corr(g[h],method='spearman'))
  a=np.array(vals); a=a[np.isfinite(a)]
  print(h,'dates',len(a),'avgN',x.groupby('date').size().mean(),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
# coverage and turnover of rank ordering
print('coverage',len(x)/((x.date.max()-x.date.min()).days*15/1),'date_range',x.date.min(),x.date.max())
rank=x.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('rank turnover',rank.diff().abs().mean().mean())
# thirds f10
for j,g in enumerate(np.array_split(x.sort_values('date').date.unique(),3)):
 a=[]
 for dt,z in x[x.date.isin(g)].groupby('date'):
  z=z.dropna(subset=['factor','f10'])
  if len(z)>=8:a.append(z.factor.corr(z.f10,method='spearman'))
 a=np.array(a);a=a[np.isfinite(a)];print('third',j+1,len(a),a.mean(),a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
# artifact
x[['date','symbol','factor']].to_csv('scripts/miner_1_20320906_momentum_acceleration_signal.csv',index=False)
