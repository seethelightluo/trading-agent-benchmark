import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}
p=pd.DataFrame(P).sort_index(); r=np.log(p).diff(); market=r.mean(axis=1); down=market<0
n=down.rolling(60,min_periods=15).sum(); cond=r.where(down).rolling(60,min_periods=15).mean(); uncond=r.rolling(60,min_periods=30).mean()
# Bayesian-shrunk downside resilience: conditional return in down days blended to unconditional, lagged.
f=(cond.mul(n/(n+10),axis=0)+uncond.mul(10/(n+10),axis=0)).shift(1)
print('DATA',p.index.min().date(),p.index.max().date(),'assets',len(P),'rows',len(p))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; vals=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(vals); print(f'h={h} dates={len(s)} meanN={np.mean(ns):.2f} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1):.6f} hit={(s>0).mean():.4f}')
print('coverage',f.notna().stack().mean(),'mean_valid',f.notna().sum(axis=1).mean(),'turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean().mean())
y=p.shift(-1)/p-1
for label,sub in [('2020-23',f.loc['2020':'2023']),('2024-27',f.loc['2024':'2027']),('2028-30',f.loc['2028':'2030']),('2031-34',f.loc['2031':'2034'])]:
 x=[]
 for d in sub.index:
  z=pd.concat([sub.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(x); print(label,'dates',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
