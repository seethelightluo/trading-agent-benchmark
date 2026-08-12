import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
    d=get_stock_daily_data(s,2200)
    if d is None or len(d)<150: d=get_index_daily_data(s,2200)
    if d is None: print('missing',s); continue
    x=d[['date','close']].copy(); x['ret']=x.close.pct_change(); x['symbol']=s; rows.append(x)
p=pd.concat(rows)
wide=p.pivot(index='date',columns='symbol',values='close').sort_index()
r=wide.pct_change(); disp=r.abs().mean(axis=1)
med=disp.rolling(60,min_periods=40).median()
# continuous dispersion multiplier, bounded to limit extreme stress concentration
vol=r.rolling(20,min_periods=12).std()*np.sqrt(20)
base=-r/vol.replace(0,np.nan)
f=base*(disp/med).clip(0.5,2.0).values[:,None]
f=f.replace([np.inf,-np.inf],np.nan)

def calc(h):
    fr=f; fut=wide.shift(-h)/wide-1
    vals=[]; dates=[]; counts=[]
    for dt in fr.index:
      a=fr.loc[dt]; b=fut.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
      if len(z)>=8:
        vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); counts.append(len(z))
    q=pd.Series(vals,index=dates).dropna()
    return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(counts)
print('cutoff',wide.index.max().date(),'dates',len(wide),'instruments',len(wide.columns))
for h in [1,3,5,10]: print('H',h,'n mean std ICIR hit avgN',calc(h))
# turnover rank signal, daily cross-section
ranks=f.rank(axis=1,pct=True); turn=(ranks-ranks.shift(1)).abs().mean(axis=1).dropna()
print('coverage',f.notna().mean().mean(),'rank_turnover',turn.mean(),'active dates',f.notna().any(axis=1).sum())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
 q=f.loc[lo:hi]; fut=wide.shift(-1)/wide-1; z=[]
 for dt in q.index:
  zz=pd.concat([q.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(zz)>=8:z.append(spearmanr(zz.iloc[:,0],zz.iloc[:,1]).statistic)
 z=pd.Series(z).dropna();print('REG',lo,hi,'n',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(len(z)) if len(z)>1 else np.nan)
out=pd.DataFrame(f.stack(),columns=['signal']).reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20261105_continuous_dispersion_reversal_signal.csv',index=False)
