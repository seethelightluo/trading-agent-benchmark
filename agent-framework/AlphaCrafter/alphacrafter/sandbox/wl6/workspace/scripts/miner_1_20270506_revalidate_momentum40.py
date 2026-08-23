import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv(Path('../persistent/stock_data')/(a+'.csv'),parse_dates=['date']).set_index('date').close.sort_index() for a in A}
p=pd.DataFrame(D).ffill().loc[:'2027-05-05']; f=p.pct_change(40).shift(1); f=f.sub(f.median(axis=1),axis=0)
def run(a,b):
 q=[]; ns=[]; ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],(p.shift(-b).div(p)-1).loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(d)
 s=pd.Series(q,index=ds); print(a,b,len(s),round(np.mean(ns),2),f'IC={s.mean():.8f}',f'ICIR={s.mean()/s.std(ddof=1)*np.sqrt(len(s)):.8f}',f'hit={(s>0).mean():.6f}')
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
  u=s.loc[lo:hi];
  if len(u): print('REG',lo,hi,len(u),f'{u.mean():.8f}',f'{u.mean()/u.std(ddof=1)*np.sqrt(len(u)):.8f}')
 print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),8))
for h in [1,5,10]:run('40d',h)
