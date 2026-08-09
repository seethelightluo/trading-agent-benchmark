import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 x=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date').query("date <= '2027-02-24'")
 r=np.log(x.close/x.close.shift(1)); vol=np.log1p(x.volume.replace(0,np.nan));
 x['factor']=r.rolling(20,min_periods=15).sum()*(vol.rolling(5,min_periods=5).mean()/vol.rolling(30,min_periods=20).mean()-1)
 x['fwd']=x.close.shift(-1)/x.close-1; x['symbol']=s; rows.append(x[['date','symbol','factor','fwd']])
a=pd.concat(rows).dropna(); obs=[]; sig=[]
for d,g in a.groupby('date'):
 if len(g)>=8:
  q=spearmanr(g.factor,g.fwd).statistic
  if np.isfinite(q): obs.append((d,q)); sig.append(g[['date','symbol','factor']])
o=pd.DataFrame(obs,columns=['date','ic']); print('dates',len(o),'avg_n',a.groupby('date').size().mean(),'IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),(o.ic>0).mean())); print(o.assign(year=o.date.dt.year).groupby('year').ic.agg(['count','mean'])); wide=a.pivot(index='date',columns='symbol',values='factor'); print('turnover',wide.rank(pct=True).diff().abs().mean().mean()); pd.concat(sig).to_csv('../persistent/factor_signals_miner_2_20270225_volconfirmed_mom.csv',index=False)
