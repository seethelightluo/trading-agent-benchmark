import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
px=pd.DataFrame({s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).sort_index().loc[:'2035-09-26']
r=px.pct_change(); ret=px.shift(1)/px.shift(121)-1
vol=r.rolling(120,min_periods=80).std().shift(1)
path=r.abs().rolling(120,min_periods=80).sum().shift(1)
eff=(ret.abs()/(path+1e-12)).clip(0,1)
# Persistent trend rewarded only when price path is efficient; volatility normalization limits crypto dominance.
f=(ret/(vol*np.sqrt(120)+0.05))*eff
f=f.replace([np.inf,-np.inf],np.nan).clip(-20,20)
def calc(h):
  vals=[]; dates=[]; ns=[]
  for i in range(len(px)-h):
    z=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
    if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(px.index[i]); ns.append(len(z))
  return pd.Series(vals,index=dates),ns
print('universe',len(U),'dates',px.index.min().date(),px.index.max().date())
for h in [5,10,20,40,60]:
 x,n=calc(h); print('H',h,'dates',len(x),'avgN',round(np.mean(n),2),'minN',min(n),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),6),'hit',round((x>0).mean(),4))
x,n=calc(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]
 if len(q): print('REG',a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6))
print('coverage',round(f.notna().sum(axis=1).div(len(U)).mean(),4),'turnover_proxy',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20350927_riskadjusted_efficiency_momentum_signal.csv',index=False)
