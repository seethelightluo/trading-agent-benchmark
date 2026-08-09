import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p): P[a]=pd.read_csv(p,parse_dates=['date']).set_index('date')['close']
p=pd.DataFrame(P).sort_index(); r=p.pct_change()
# range-position persistence: close location in trailing 40d high-low, lagged one day
hi=p.rolling(40,min_periods=25).max(); lo=p.rolling(40,min_periods=25).min(); f=((p-lo)/(hi-lo)).shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; out=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(out); print(f'h={h} dates={len(s)} meanN={np.mean(ns):.2f} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1):.6f} hit={(s>0).mean():.4f}')
print('coverage',f.notna().stack().mean(),'mean_valid',f.notna().sum(axis=1).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for label,sub in [('2025-27',f.loc['2025':'2027']),('2028-30',f.loc['2028':'2030-04-18']),('latest120',f.iloc[-120:])]:
 y=p.shift(-10)/p-1; x=[]
 for d in sub.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(label,'n',len(x),'IC',np.mean(x),'ICIR',np.mean(x)/np.std(x,ddof=1))
