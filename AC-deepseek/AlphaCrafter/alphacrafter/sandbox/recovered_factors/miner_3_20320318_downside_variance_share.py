import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# downside asymmetry: fraction of realized variance from negative daily returns, smoothed 40d; lower downside share potentially resilience
neg=(r.clip(upper=0)**2).rolling(40,min_periods=25).mean()
tot=(r**2).rolling(40,min_periods=25).mean()
f=-(neg/tot) # higher = less downside variance
# orthogonal-ish normalize by total vol? test downside share directly
for h in [1,5,10,20]:
  fr=p.shift(-h)/p-1
  vals=[]; dates=[]
  for d in f.index:
    x=f.loc[d]; y=fr.loc[d]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
      vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(d)
  a=np.array(vals); print('H',h,'dates',len(a),'meanIC',a.mean(),'std',a.std(ddof=1),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'nmean',np.mean([f.loc[d].notna().sum() for d in dates]))
# regimes and turnover
for name,sl in [('2020-23',slice('2020','2023')),('2024-27',slice('2024','2027')),('2028-30',slice('2028','2030')),('2031+',slice('2031',None)),('recent120',slice(None,None))]:
 ff=f.loc[sl]; yy=(p.shift(-1)/p-1).loc[sl]; a=[]
 for d in ff.index:
  z=pd.concat([ff.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a[-120:] if name=='recent120' else a); print(name,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6) if len(a)>1 else 0)
print('coverage',f.notna().mean().mean(),'turnover10', (f.rank(axis=1,pct=True).diff(10).abs().mean().mean()))
