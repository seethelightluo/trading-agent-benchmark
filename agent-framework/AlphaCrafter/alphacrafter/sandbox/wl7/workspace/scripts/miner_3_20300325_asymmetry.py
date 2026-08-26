import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close']
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# upside/downside asymmetry: positive-return contribution relative to downside, scaled by total activity
up=r.clip(lower=0).rolling(20).sum(); dn=(-r.clip(upper=0)).rolling(20).sum()
# reward assets with upside dominance, but penalize extreme total movement; lagged
f=((up-dn)/(up+dn+1e-12) / (r.rolling(20).std()+1e-12)).shift(1)
rows=[]
for i in range(len(P)-10):
 dt=P.index[i]; fut=P.iloc[i+1:i+11].iloc[-1]/P.iloc[i]-1
 x=f.iloc[i]
 ok=x.notna()&fut.notna()
 if ok.sum()>=8:
  rows.append((dt,spearmanr(x[ok],fut[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for h in [1,5,10,20,40]:
 ff=P.shift(-h)/P-1; rr=[]
 for i in range(len(P)-h):
  ok=f.iloc[i].notna()&ff.iloc[i].notna()
  if ok.sum()>=8: rr.append(spearmanr(f.iloc[i][ok],ff.iloc[i][ok]).statistic)
 print('decay',h, np.nanmean(rr), np.nanmean(rr)/np.nanstd(rr,ddof=1),len(rr))
print('10d',z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),'dates',len(z),'avgN',z.n.mean())
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-03-20')]:
 q=z.loc[a:b]; print(a,b,q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),len(q))
# turnover rank
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean().mean())
z.to_csv('scripts/miner_3_20300325_asymmetry_ic.csv'); f.to_csv('scripts/miner_3_20300325_asymmetry_signal.csv')
