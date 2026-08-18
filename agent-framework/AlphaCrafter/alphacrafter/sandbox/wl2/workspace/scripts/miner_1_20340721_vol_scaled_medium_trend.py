import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
p=pd.DataFrame(D).sort_index().ffill().loc[:'2034-07-20']
r=p.pct_change()
# volatility-scaled medium-term trend: trailing 20d return divided by trailing 20d realized vol, lagged
f=(p.pct_change(20)/r.rolling(20).std()).shift(1)
rows=[]
for h in [5,10,20]:
  ic=[]
  for dt in f.index:
    x=f.loc[dt]; y=p.pct_change(h).shift(-h).loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.array(ic); rows.append((h,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
# turnover average rank changes
rank=f.rank(axis=1,pct=True); turnover=(rank.diff().abs().mean(axis=1)).mean()
print('dates',len(p),'assets',len(U),'valid',f.notna().sum(axis=1).mean(),'coverage',f.notna().mean().mean(),'turnover',turnover)
for x in rows: print('horizon dates IC ICIR hit',x)
print('period',p.index.min().date(),p.index.max().date())
# regimes half periods
for name,ix in [('early',f.index<'2025-01-01'),('mid',(f.index>='2025-01-01')&(f.index<'2030-01-01')),('recent',f.index>='2030-01-01')]:
 a=[]
 for dt in f.index[ix]:
  z=pd.concat([f.loc[dt],p.pct_change(10).shift(-10).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print(name,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1) if len(a)>1 else np.nan)
