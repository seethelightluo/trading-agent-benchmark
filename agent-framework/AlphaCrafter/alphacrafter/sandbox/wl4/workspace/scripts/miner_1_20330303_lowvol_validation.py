import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
from scipy.stats import spearmanr
W=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in W:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=4000)
 if d is None:continue
 d=d.sort_values('date').set_index('date'); c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change()
 # low volatility with a mild positive risk-adjusted trend; all inputs lagged
 vol=r.rolling(40).std().shift(1); mom=c.pct_change(60).shift(1)
 f=(-np.log(vol+1e-12)+0.25*mom/(r.rolling(60).std().shift(1)*np.sqrt(60)+1e-12))
 rows.append(pd.DataFrame({'symbol':s,'factor':f,'fwd10':c.shift(-10)/c-1}))
x=pd.concat(rows).reset_index().dropna(); obs=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>2:obs.append((dt,spearmanr(g.factor,g.fwd10).statistic,len(g)))
o=pd.DataFrame(obs,columns=['date','ic','n']);print('dates',len(o),'avgN',o.n.mean(),'IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(ddof=1)*np.sqrt(252),'hit',(o.ic>0).mean())
for n in [365,730,1095]:
 z=o[o.date>=o.date.max()-pd.Timedelta(days=n)];print('recent',n,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252))
print('turnover',x.pivot_table(index='date',columns='symbol',values='factor').rank(pct=True).diff().abs().mean(axis=1).mean());print('period',o.date.min(),o.date.max())
x.to_csv('scripts/artifacts/miner_1_20330303_lowvol_signal.csv',index=False)
