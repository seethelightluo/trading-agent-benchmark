import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
px=pd.DataFrame({s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).sort_index().loc[:'2035-10-10']
r=px.pct_change(); r5=px.shift(1)/px.shift(6)-1; v=r.rolling(20,min_periods=12).std().shift(1)
# Short reversal normalized by local risk; only emphasize when the market cross-section is unusually dispersed.
base=-r5/(v*np.sqrt(5)+.03)
disp=r.shift(1).rolling(20,min_periods=12).std().mean(axis=1)
gate=(disp/disp.rolling(120,min_periods=60).median()).clip(.5,2.0)
f=(base.mul(gate,axis=0)).replace([np.inf,-np.inf],np.nan).clip(-20,20)
def calc(h):
 vals=[]; dates=[]; ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(px.index[i]); ns.append(len(z))
 x=pd.Series(vals,index=dates); return x,ns
print('universe',len(U),'dates',px.index.min().date(),px.index.max().date())
for h in [5,10,20,40,60]:
 x,n=calc(h); print('H',h,'dates',len(x),'avgN',round(np.mean(n),2),'minN',min(n),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),6),'hit',round((x>0).mean(),4))
x,n=calc(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]
 if len(q): print('REG',a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6),round((q>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).div(len(U)).mean(),4),'turnover_proxy',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20351011_dispersion_regime_reversal_signal.csv',index=False)
