import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
d={s:ld(s) for s in U}; c=pd.concat({s:x.close for s,x in d.items()},axis=1); r=np.log(c).diff(); v=r.rolling(20,min_periods=12).std(); z=(-r/v)
# Extreme downside shock reversal, with a continuous score only below the lagged volatility threshold.
f=z.where(z>1.5,0).clip(upper=4).shift(1)
rows=[]
for dt in r.index:
 a=f.loc[dt]; y=r.shift(-1).loc[dt]; ok=a.notna()&y.notna()&(a!=0)
 if ok.sum()>=8: rows.append((dt,spearmanr(a[ok],y[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(x),'assets',15,'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean(),'avg_n',x.n.mean())
for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-08-04')]:
 q=x.loc[a:b]; print('regime',a,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
print('turnover_proxy',np.mean(np.sign(f).diff().abs().stack()>0)); f.to_csv('scripts/miner_3_20330805_extreme_shock_reversal_signal.csv')
