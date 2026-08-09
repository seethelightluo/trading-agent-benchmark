import pandas as pd, numpy as np
from scipy.stats import spearmanr
import glob
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 px[a]=d['close'].astype(float)
df=pd.DataFrame(px).sort_index()
r=df.pct_change()
# candidate: negative downside/upside realized-vol ratio, 60-day rolling, lagged one day
up=r.where(r>0); dn=r.where(r<0)
uv=up.rolling(60,min_periods=30).std(); dv=dn.rolling(60,min_periods=30).std()
f=-(dv/(uv+1e-8)).shift(1)
# forward close-to-close returns, with non-synchronous missing values handled cross-sectionally
for h in [1,5,10,20]:
  fr=df.shift(-h)/df-1
  vals=[]; dates=[]; ns=[]
  for t in df.index:
    x=f.loc[t]; y=fr.loc[t]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
      vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(t); ns.append(len(z))
  a=np.array(vals); mean=a.mean(); sd=a.std(ddof=1)
  print(f'H{h}: dates={len(a)} meanN={np.mean(ns):.2f} IC={mean:.6f} ICIR={mean/sd*np.sqrt(1) if sd else np.nan:.6f} hit={np.mean(a>0):.4f}')
  for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31')]:
   q=np.array([v for v,t in zip(vals,dates) if pd.Timestamp(lo)<=t<=pd.Timestamp(hi)])
   print(' ',lo,hi,'n',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# coverage and turnover
print('coverage',f.notna().mean().mean(),'mean valid',f.notna().sum(axis=1).mean())
print('10d turnover',((f.rank(axis=1,pct=True)-f.shift(10).rank(axis=1,pct=True)).abs().mean(axis=1).dropna().mean()))
