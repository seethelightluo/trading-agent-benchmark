import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
W=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in W:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<250:d=get_index_daily_data(s,days=4000)
 if d is None or len(d)<250:continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date');c=pd.to_numeric(d.close,errors='coerce'); lo=c.rolling(120).min().shift(1);hi=c.rolling(120).max().shift(1)
 f=(1-(c-lo)/(hi-lo+1e-12)).shift(0); rows.append(pd.DataFrame({'symbol':s,'factor':f,'fwd10':c.shift(-10)/c-1}))
x=pd.concat(rows).reset_index().dropna();obs=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>2:obs.append((dt,spearmanr(g.factor,g.fwd10).statistic,len(g)))
o=pd.DataFrame(obs,columns=['date','ic','n']).sort_values('date');print('dates',len(o),'avgN',o.n.mean(),'instruments',len(rows),'coverage',len(x)/len(pd.concat(rows).reset_index()));print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1)*np.sqrt(252),(o.ic>0).mean()))
for n in [365,730,1095]:
 z=o[o.date>=o.date.max()-pd.Timedelta(days=n)];print('recent',n,'dates',len(z),'IC %.6f ICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252)))
wide=x.pivot_table(index='date',columns='symbol',values='factor');print('turnover',wide.rank(pct=True).diff().abs().mean(axis=1).mean(),'period',o.date.min(),o.date.max());x.to_csv('scripts/artifacts/miner_1_20330317_range_reversal_revalidation_signal.csv',index=False)
