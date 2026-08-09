import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}
p=pd.DataFrame(P).sort_index(); r=np.log(p).diff()
# Three-session cross-sectional reversal, damped by recent absolute-return activity. Lag one day.
ret3=r.rolling(3,min_periods=3).sum(); activity=r.abs().rolling(20,min_periods=10).sum().clip(lower=0.02)
f=(-ret3/activity).shift(1); f=f.sub(f.mean(axis=1),axis=0)
print('DATA',p.index.min(),p.index.max(),'assets',len(P),'rows',len(p))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(vals);print(f'h={h} dates={len(s)} meanN={np.mean(ns):.2f} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1):.6f} hit={(s>0).mean():.4f}')
print('coverage',f.notna().stack().mean(),'mean_valid',f.notna().sum(axis=1).mean(),'turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean().mean())
y=p.shift(-1)/p-1
for label,sub in [('2024-27',f.loc['2024':'2027']),('2028-30',f.loc['2028':'2030']),('2031-34',f.loc['2031':'2034'])]:
 x=[]
 for d in sub.index:
  z=pd.concat([sub.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(x);print(label,'dates',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1))
