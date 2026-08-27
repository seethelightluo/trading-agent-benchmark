import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
cl={s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}
p=pd.DataFrame(cl).sort_index().loc[:'2035-07-18']; r=p.pct_change(); vol=r.rolling(60,min_periods=40).std()
trough=p.rolling(120,min_periods=80).min()
# Prefer assets still materially below their 120-session trough-adjusted recovery level, scaled by volatility.
f=((1-p/trough)/(vol+0.005)).replace([np.inf,-np.inf],np.nan).shift(1).clip(-20,20)
def calc(h):
 vals=[]; ds=[]; ns=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(p.index[i]); ns.append(len(z))
 x=pd.Series(vals,index=pd.DatetimeIndex(ds)); return x,np.mean(ns)
print('universe',len(U),'dates',p.index.min().date(),p.index.max().date())
for h in [5,10,20,40,60]:
 x,an=calc(h); print(f'H{h} dates {len(x)} avgN {an:.2f} IC {x.mean():.6f} ICIR {x.mean()/x.std(ddof=1)*np.sqrt(252):.6f} hit {(x>0).mean():.4f}')
x,_=calc(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]
 if len(q): print(f'REG {a}-{b} dates {len(q)} IC {q.mean():.6f} ICIR {q.mean()/q.std(ddof=1)*np.sqrt(252):.6f}')
print('coverage',f.notna().sum(axis=1).div(15).mean(),'turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20350719_drawdown_reversal_signal.csv',index=False)
