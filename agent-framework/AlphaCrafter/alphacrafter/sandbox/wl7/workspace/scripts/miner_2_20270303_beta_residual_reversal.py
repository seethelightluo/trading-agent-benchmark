import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(s):
    
    try: d=get_index_daily_data(s, days=3000)
    except FileNotFoundError: d=get_stock_daily_data(s, days=3000)
    if d is None or len(d)==0:return pd.Series(dtype=float)
    x=d.copy(); x['date']=pd.to_datetime(x['date']); return x.set_index('date')['close'].astype(float)
px=pd.concat({s:load(s) for s in U},axis=1).sort_index().ffill()
r=np.log(px).diff()
# market return and residual 5d return, using only history through t; lag factor one day
m=r.mean(axis=1)
# rolling beta based on 60 days, then residual cumulative 5d
cov=r.rolling(60,min_periods=30).cov(m)
var=m.rolling(60,min_periods=30).var()
beta=cov.div(var.replace(0,np.nan),axis=0)
res5=(r.rolling(5).sum()).sub(beta.mul(m.rolling(5).sum(),axis=0))
vol=r.rolling(20).std()
f=(-res5/vol).shift(1)
f=f.replace([np.inf,-np.inf],np.nan)
fr=r.shift(-1)
ics=[]; rows=[]
for dt in f.index:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1]); ics.append(ic); rows.append((dt,ic,len(z)))
ser=pd.Series(ics,index=[x[0] for x in rows])
print('dates',len(ser),'avg_n',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15)
print('IC %.8f ICIR %.8f hit %.4f turnover %.4f'%(ser.mean(),ser.mean()/ser.std()*np.sqrt(252), (ser>0).mean(), f.rank(axis=1,pct=True).diff().abs().mean().mean()))
for h in [5,10,20]:
  ff=r.shift(-h).rolling(h).sum().shift(-(h-1)) # return t+1..t+h approximately
  a=[]
  for dt in f.index:
    z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
    if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
  print('decay',h,np.nanmean(a),len(a))
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=ser[(ser.index>=a)&(ser.index<=b+'-12-31')];print('regime',a,b,len(q),q.mean())
# artifact
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270303_beta_residual_reversal_signal.csv',index=False)
