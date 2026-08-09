import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets}
p=pd.DataFrame(P).sort_index().loc[:'2034-07-18']; r=p.pct_change()
# Liquidity-free short-term reversal: recent 5-session return, scaled by trailing 20-session realized volatility.
f=-p.pct_change(5)/(r.rolling(20).std()*np.sqrt(20)+1e-8)
print('candidate=volatility_normalized_5d_reversal cutoff',p.index.max().date(),'rows',len(p),'assets',len(assets))
for h in [1,5,10,20]:
 vals=[]; ns=[]; dates=[]
 fr=p.shift(-h)/p-1
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
 v=np.asarray(vals); print('H',h,'dates',len(v),'meanN',round(np.mean(ns),2),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round(np.mean(v>0),4))
 print('  regimes',[(yr,round(np.mean(v[np.array([d.year//5*5==yr for d in dates])]),6),int(sum(d.year//5*5==yr))) for yr in [2020,2025,2030] if sum(d.year//5*5==yr)>0])
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
print('library_audit FAILED: no exact common-cell reconstruction performed')
