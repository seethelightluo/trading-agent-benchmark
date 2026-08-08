import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p): D[a]=pd.read_csv(p,parse_dates=['date']).set_index('date')['close']
prices=pd.DataFrame(D).sort_index()
r=prices.pct_change()
# path efficiency: signed net move / total absolute daily path, lag one day
net=prices/prices.shift(20)-1
path=r.abs().rolling(20,min_periods=14).sum()
f=(net/path).shift(1)
# forward arithmetic returns
for h in [1,5,10,20]:
 fr=prices.shift(-h)/prices-1
 vals=[]; dates=[]; ns=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 s=pd.Series(vals,index=dates).dropna()
 print(f'h={h} dates={len(s)} meanN={np.mean(ns):.2f} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1):.6f} hit={np.mean(s>0):.4f}')
print('coverage',f.notna().stack().mean(),'mean daily valid',f.notna().sum(axis=1).mean())
# turnover rank proxy
ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean().mean())
# regime and latest h10
fr=prices.shift(-10)/prices-1
for label,lo,hi in [('2020-24','2020','2024-12-31'),('2025-27','2025','2027-12-31'),('2028-30','2028','2030-04-18'),('latest120','2030-??','2030-04-18')]:
 if label=='latest120': sub=f.index[-120:]
 else: sub=f.loc[lo:hi].index
 a=[]
 for dt in sub:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(label,len(a),np.mean(a) if a else np.nan,(np.mean(a)/np.std(a,ddof=1)) if len(a)>1 else np.nan)
