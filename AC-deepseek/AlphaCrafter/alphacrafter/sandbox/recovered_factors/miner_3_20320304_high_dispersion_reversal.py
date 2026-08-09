import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); p[a]=d.set_index('date').close
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
# High-dispersion conditioned normalized reversal: fade recent 5d move when
# 20d cross-sectional return dispersion is in its trailing 120d upper quartile.
cs_disp=r.std(axis=1).rolling(20,min_periods=15).mean()
threshold=cs_disp.rolling(120,min_periods=80).quantile(.75)
active=(cs_disp>=threshold)
vol=r.rolling(20,min_periods=15).std()
fac=(-(p.pct_change(5))/(vol*np.sqrt(5))).where(active, np.nan)
print('active_fraction=%.4f'%active.mean())
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in p.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); print(f'H{h} dates={len(a)} meanN={np.mean(ns):.2f} IC={a.mean():.6f} ICIR={a.mean()/(a.std(ddof=1)+1e-12):.6f} hit={np.mean(a>0):.3f}')
ranks=fac.rank(axis=1,pct=True); turns=[]
for i in range(10,len(ranks),10):
 a=ranks.iloc[i-10]; b=ranks.iloc[i]; c=a.dropna().index.intersection(b.dropna()).intersection(ranks.iloc[i].dropna().index)
 if len(c)>=8: turns.append(np.mean(abs(a[c]-b[c])))
print('coverage=%.4f dates=%d assets=%d turnover10=%.4f'%(fac.notna().sum().sum()/fac.size,len(p),len(assets),np.mean(turns)))
for label,sub in [('2020-23',p.index[p.index.year<=2023]),('2024-27',p.index[(p.index.year>=2024)&(p.index.year<=2027)]),('2028-30',p.index[(p.index.year>=2028)&(p.index.year<=2030)]),('2031+',p.index[p.index.year>=2031]),('recent120',p.index[-120:])]:
 a=[]
 for dt in sub:
  z=pd.concat([fac.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a); print(label,'n=',len(a),'IC=%.6f ICIR=%.6f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12)))
