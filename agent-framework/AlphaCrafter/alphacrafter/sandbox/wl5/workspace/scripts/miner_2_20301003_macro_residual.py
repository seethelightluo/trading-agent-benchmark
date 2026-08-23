import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym,days=4000):
 d=get_stock_daily_data(sym,days)
 if d is None or len(d)==0:return pd.Series(dtype=float)
 return d.set_index('date')['close'].astype(float).sort_index()
px=pd.DataFrame({s:load(s) for s in ASSETS}).loc[:pd.Timestamp('2030-10-02')]
ret=px.pct_change(); bench=ret['SPX']; mom=px/px.shift(20)-1
cov=ret.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
resid=mom-beta.mul(mom['SPX'],axis=0); vol=ret.rolling(60,min_periods=40).std()*np.sqrt(252)
sig=-(resid/vol); sig.to_csv('scripts/miner_2_20301003_macro_residual_reversal_signal.csv',index_label='date')
fwd=px.shift(-10)/px-1; obs=[]; turnovers=[]; prev=None
for dt in sig.index:
 x=sig.loc[dt].replace([np.inf,-np.inf],np.nan).dropna(); y=fwd.loc[dt].reindex(x.index).dropna(); x=x.reindex(y.index)
 if len(x)<8:continue
 obs.append((dt,x.corr(y,method='spearman'),len(x))); ranks=x.rank(pct=True)
 if prev is not None:turnovers.append((ranks-prev).abs().mean())
 prev=ranks
z=pd.DataFrame(obs,columns=['date','ic','n']).dropna(); mean=z.ic.mean(); sd=z.ic.std(ddof=1)
print('factor macro_beta_residual_reversal_20d'); print('dates',len(z),'mean_n',z.n.mean(),'coverage',z.n.mean()/15,'IC',mean,'ICIR',mean/sd*np.sqrt(252),'hit',(z.ic>0).mean(),'turnover',np.mean(turnovers))
for name,a,b in [('2020-24','2020-01-01','2024-12-31'),('2025-27','2025-01-01','2027-12-31'),('2028-29','2028-01-01','2029-12-31'),('2030','2030-01-01','2030-10-02')]:
 q=z[(z.date>=a)&(z.date<=b)]; print(name,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
