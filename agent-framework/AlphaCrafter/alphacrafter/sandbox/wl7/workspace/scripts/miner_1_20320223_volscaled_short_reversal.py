import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-02-22')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
close=pd.DataFrame(px).sort_index().loc[:cut]; rets=close.pct_change()
vol=rets.rolling(20,min_periods=15).std()*np.sqrt(20)
r5=close.pct_change(5); r60=close.pct_change(60)
f=(-r5/vol) * (1-0.35*np.sign(r60)*np.sign(r5))
for k in [1,5,10,20]:
 vals=[]
 for d in close.index:
  x=f.loc[d]; y=close.shift(-k).loc[d]/close.loc[d]-1
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: vals.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); mean=q.ic.mean(); sd=q.ic.std(ddof=1)
 # annualized ICIR convention used by prior miner: daily mean/std * sqrt(252/k)
 icir=mean/sd*np.sqrt(252/k)
 print(k,'dates',len(q),'avgN',q.n.mean(),'IC',round(mean,6),'ICIR',round(icir,6),'hit',round((q.ic>0).mean(),4))
rank=f.rank(axis=1,pct=True); print('coverage',round(f.notna().sum(axis=1).mean()/15,6),'turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6),'span',close.index.min().date(),close.index.max().date())
f.to_csv('scripts/miner_1_20320223_volscaled_short_reversal_signal.csv',index_label='date')
