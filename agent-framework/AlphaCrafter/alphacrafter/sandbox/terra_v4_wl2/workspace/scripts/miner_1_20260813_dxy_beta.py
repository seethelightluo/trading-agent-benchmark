import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def rd(s,p): return pd.read_csv(p+s+'.csv',parse_dates=['date']).set_index('date').close.pct_change()
R=pd.concat([rd(s,'../persistent/stock_data/') for s in U],axis=1);R.columns=U
m=rd('DXY','../persistent/index_data/'); R=R.join(m.rename('DXY'),how='inner').loc[:'2026-08-13'];m=R.pop('DXY')
b=R.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var(),axis=0);f=-b
out={h:[] for h in [1,5,10]}; dates={h:[] for h in out}
for i,dt in enumerate(R.index):
 for h in out:
  y=R.iloc[i+1:i+1+h].sum();x=f.iloc[i];ok=x.notna()&y.notna()&np.isfinite(x)&np.isfinite(y)
  if ok.sum()>=8:
   z=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(z):out[h].append(z);dates[h].append(dt)
def stat(a):
 a=np.array(a);return len(a),round(float(a.mean()),5),round(float(a.mean()/a.std(ddof=1)),5),round(float((a>0).mean()),4)
for h in out: print(h,stat(out[h]),'period',dates[h][0],dates[h][-1])
print('names',round(R.notna().sum(axis=1).mean(),2),'coverage',round(f.notna().sum().sum()/f.size,4))
for yy in range(2020,2027):
 a=[v for d,v in zip(dates[1],out[1]) if d.year==yy];print(yy,stat(a) if a else None)
print('rho mom',pd.concat([f.stack(),R.rolling(20).mean().stack()],axis=1).corr().iloc[0,1]);print('rho rev',pd.concat([f.stack(),(-R.rolling(5).mean()).stack()],axis=1).corr().iloc[0,1])
