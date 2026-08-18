import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
u=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
f={}
for s in u:
 d=get_stock_daily_data(s,days=6000)
 if d is None: d=get_index_daily_data(s,days=6000)
 if d is not None: f[s]=d.set_index('date')['close'].astype(float)
p=pd.concat(f,axis=1).sort_index().ffill(); lp=np.log(p); r=lp.diff()
# acceleration: short trend relative to medium trend, scaled by recent volatility
fac=((lp-lp.shift(10))-(lp-lp.shift(40))*.25)/(r.rolling(20).std()*np.sqrt(252)+1e-8)
y=p.shift(-10)/p-1
rows=[]
for t in fac.index:
 z=pd.concat([fac.loc[t],y.loc[t]],axis=1).dropna()
 if len(z)>=8: rows.append((t,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
i=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); rank=fac.rank(axis=1,pct=True)
print('candidate=volscaled_trend_acceleration_10_40');print('dates=%d instruments=%d coverage=%.4f turnover=%.6f'%(len(i),len(p.columns),i.n.mean()/len(p.columns),rank.diff().abs().mean(axis=1).mean()))
for q,a in [('full',i),('120',i.tail(120)),('252',i.tail(252)),('756',i.tail(756)),('1260',i.tail(1260))]: print(q,'IC=%.8f ICIR=%.8f hit=%.4f n=%d'%(a.ic.mean(),a.ic.mean()/(a.ic.std(ddof=1)+1e-12),(a.ic>0).mean(),len(a)))
for h in [5,10,20,40]:
 yy=p.shift(-h)/p-1; a=[]
 for t in fac.index:
  z=pd.concat([fac.loc[t],yy.loc[t]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(a);print('decay',h,'IC=%.8f ICIR=%.8f n=%d'%(np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),len(a)))
fac.to_csv('scripts/miner_1_20340120_accel_signal.csv');i.to_csv('scripts/miner_1_20340120_accel_ic.csv')
