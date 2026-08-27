import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
data={}
for s in U:
 d=pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()
 data[s]=d[['open','close']]
D=pd.concat({s:x for s,x in data.items()},axis=1).sort_index().loc[:'2035-08-29']
o=D.xs('open',axis=1,level=1); c=D.xs('close',axis=1,level=1)
# One-session intraday reversal, lagged to avoid using current completed session in its own forecast.
f=-(c/o-1).replace([np.inf,-np.inf],np.nan).shift(1)
# Winsorize cross-sectionally to limit isolated synthetic OHLC errors.
f=f.clip(f.quantile(.02,axis=1),f.quantile(.98,axis=1),axis=0)
def calc(h):
 vals=[]; dates=[]; ns=[]
 fr=c.pct_change(h).shift(-h)
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 x=pd.Series(vals,index=pd.DatetimeIndex(dates)); return x,np.mean(ns)
print('universe',len(U),'period',c.index.min().date(),c.index.max().date())
for h in [1,5,10,20,40]:
 x,an=calc(h); print(f'H{h} dates {len(x)} avgN {an:.2f} IC {x.mean():.6f} ICIR {x.mean()/x.std(ddof=1)*np.sqrt(252):.6f} hit {(x>0).mean():.4f}')
x,_=calc(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]
 if len(q): print(f'REG {a}-{b} dates {len(q)} IC {q.mean():.6f} ICIR {q.mean()/q.std(ddof=1)*np.sqrt(252):.6f}')
print('coverage',f.notna().sum(axis=1).div(15).mean(),'turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20350830_intraday_reversal_signal.csv',index=False)
