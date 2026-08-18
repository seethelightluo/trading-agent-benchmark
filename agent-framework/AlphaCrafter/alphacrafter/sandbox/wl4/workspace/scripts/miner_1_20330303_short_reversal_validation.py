import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
from scipy.stats import spearmanr

watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in watch:
    df=get_stock_daily_data(s, days=4000)
    if df is None or len(df)<120: df=get_index_daily_data(s, days=4000)
    if df is None or len(df)<120: continue
    d=df.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
    c=pd.to_numeric(d['close'], errors='coerce'); r=c.pct_change()
    # lagged 5-day reversal, scaled by lagged 20-day realized volatility
    f=-(c.pct_change(5).shift(1))/(r.rolling(20).std().shift(1)*np.sqrt(5)+1e-12)
    rows.append(pd.DataFrame({'symbol':s,'factor':f,'fwd10':c.shift(-10)/c-1}))
x=pd.concat(rows).reset_index().dropna()
obs=[]
for dt,g in x.groupby('date'):
    if len(g)>=8 and g['factor'].nunique()>2 and g['fwd10'].nunique()>2:
        ic=spearmanr(g.factor,g.fwd10).statistic
        obs.append((dt,ic,len(g)))
o=pd.DataFrame(obs,columns=['date','ic','n']).sort_values('date')
print('dates',len(o),'avgN',o.n.mean(),'coverage',len(x)/sum(len(pd.concat(rows).loc[d]) for d in pd.concat(rows).index.unique()))
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/(o.ic.std(ddof=1)+1e-12)*np.sqrt(252), (o.ic>0).mean()))
for n in [365,730,1095]:
 z=o[o.date>=o.date.max()-pd.Timedelta(days=n)]
 print('recent',n,'dates',len(z),'IC %.6f ICIR %.6f'%(z.ic.mean(),z.ic.mean()/(z.ic.std(ddof=1)+1e-12)*np.sqrt(252)))
# turnover: rank ordering changes over consecutive observations
wide=x.pivot_table(index='date',columns='symbol',values='factor')
ranks=wide.rank(pct=True); turn=ranks.diff().abs().mean(axis=1).mean()
print('turnover',turn,'period',o.date.min(),o.date.max())
# artifact
x.to_csv('scripts/artifacts/miner_1_20330303_short_reversal_signal.csv',index=False)
