import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in watch:
    d=get_stock_daily_data(s,days=4000)
    if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
    if d is None or len(d)<150: continue
    d=d.copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
    c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change()
    down=r.where(r<0,0).rolling(30).std().shift(1)
    # lagged 30d trend rewarded, penalized only by downside risk
    f=c.pct_change(30).shift(1)/(down*np.sqrt(252)+1e-12)
    rows.append(pd.DataFrame({'symbol':s,'factor':f,'fwd10':c.shift(-10)/c-1}))
x=pd.concat(rows).reset_index(); base=x.dropna()
obs=[]
for dt,g in base.groupby('date'):
    if len(g)>=8 and g.factor.nunique()>2 and g.fwd10.nunique()>2:
        obs.append((dt,spearmanr(g.factor,g.fwd10).statistic,len(g)))
o=pd.DataFrame(obs,columns=['date','ic','n']).sort_values('date')
print('dates',len(o),'avgN',round(o.n.mean(),2),'instruments',len(rows),'coverage',round(len(base)/len(x),4))
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/(o.ic.std(ddof=1)+1e-12)*np.sqrt(252),(o.ic>0).mean()))
for n in [365,730,1095]:
 z=o[o.date>=o.date.max()-pd.Timedelta(days=n)]
 print('recent',n,'dates',len(z),'IC %.6f ICIR %.6f'%(z.ic.mean(),z.ic.mean()/(z.ic.std(ddof=1)+1e-12)*np.sqrt(252)))
wide=base.pivot_table(index='date',columns='symbol',values='factor'); turn=wide.rank(pct=True).diff().abs().mean(axis=1).mean()
print('turnover %.6f period %s %s'%(turn,o.date.min().date(),o.date.max().date()))
base.to_csv('scripts/artifacts/miner_1_20330317_downside_adjusted_momentum_signal.csv',index=False)
