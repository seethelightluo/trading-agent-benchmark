import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Downside-adjusted relative reversal: recent 5d cross-sectional residual, divided by downside deviation;
# negative residual favors short-term mean reversion, while downside scaling penalizes fragile assets.
raw=r.rolling(5,min_periods=4).sum(); med=raw.median(axis=1); resid=raw.sub(med,axis=0)
down=r.where(r<0).rolling(20,min_periods=10).std()
f=(-resid/(down+1e-8)).shift(1)
frs={h:p.shift(-h)/p-1 for h in [5,10,20]}
def calc(fwd, dates=None):
 vals=[]; ns=[]
 ix=f.index if dates is None else f.index.intersection(dates)
 for dt in ix:
  z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 x=pd.Series(vals).dropna(); return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()
print('period',p.index.min().date(),p.index.max().date(),'instruments',len(U))
for h,fw in frs.items(): print('H',h,'dates %.0f avgN %.2f IC %.6f ICIR %.6f hit %.4f'%calc(fw))
print('coverage',f.notna().sum().sum()/p.notna().sum().sum(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for n in [365,730,1095]:
 d=f.index[-n:]; print('recent',n,'H10 dates %.0f avgN %.2f IC %.6f ICIR %.6f hit %.4f'%calc(frs[10],d))
