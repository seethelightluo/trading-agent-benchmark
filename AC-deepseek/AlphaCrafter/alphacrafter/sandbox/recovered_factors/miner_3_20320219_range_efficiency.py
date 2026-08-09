import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}; vol={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); q=d.set_index('date'); p[a]=q.close; vol[a]=q.volume
p=pd.DataFrame(p).sort_index(); vol=pd.DataFrame(vol).reindex(p.index)
r=p.pct_change()
# Signed range efficiency: net 40d return divided by cumulative absolute daily returns.
fac=r.rolling(40).sum()/r.abs().rolling(40).sum()
# Require meaningful activity but no future information; volume is only a validity filter.
fac=fac.where(vol.rolling(20).mean()>0)
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in p.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); print(f'H{h} dates={len(a)} IC={a.mean():.6f} ICIR={a.mean()/(a.std(ddof=1)+1e-12):.6f} hit={np.mean(a>0):.3f} meanN={np.mean(ns):.2f}')
ranks=fac.rank(axis=1,pct=True); turns=[]
for i in range(10,len(ranks),10):
 a=ranks.iloc[i-10]; b=ranks.iloc[i]; c=a.dropna().index.intersection(b.dropna().index)
 if len(c)>=8: turns.append(np.mean(abs(a[c]-b[c])))
print('coverage=%.4f dates=%d assets=%d turnover10=%.4f'%(fac.notna().sum().sum()/fac.size,len(p),len(assets),np.mean(turns)))
fwd=p.shift(-1)/p-1
for label,sub in [('2020-23',p.index[p.index.year<=2023]),('2024-27',p.index[(p.index.year>=2024)&(p.index.year<=2027)]),('2028-30',p.index[(p.index.year>=2028)&(p.index.year<=2030)]),('2031+',p.index[p.index.year>=2031]),('recent120',p.index[-120:])]:
 a=[]
 for dt in sub:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a); print(label,'n=',len(a),'IC=%.6f ICIR=%.6f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12)))
