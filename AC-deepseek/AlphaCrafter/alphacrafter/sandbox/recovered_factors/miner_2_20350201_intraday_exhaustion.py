import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut='2035-01-31'
D={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 D[a]=d.loc[:cut,['open','close','high','low']]
# Intraday path exhaustion: fade the signed open-to-close move, normalized by true intraday range.
O=pd.DataFrame({a:D[a]['open'] for a in assets}); C=pd.DataFrame({a:D[a]['close'] for a in assets})
H=pd.DataFrame({a:D[a]['high'] for a in assets}); L=pd.DataFrame({a:D[a]['low'] for a in assets})
rng=(H-L).replace(0,np.nan)
F=-(C/O-1)/((rng/O).rolling(5,min_periods=3).mean())
# robust clipping cross-section, not future-looking
F=F.clip(-5,5)
for h in [1,5,10,20]:
 fw=C.shift(-h)/C-1; vals=[]; ns=[]; dates=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
 s=pd.Series(vals,index=dates)
 print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
 for label,mask in [('2025-29',(s.index.year>=2025)&(s.index.year<=2029)),('2030-32',(s.index.year>=2030)&(s.index.year<=2032)),('2033-35',s.index.year>=2033)]:
  q=s[mask]; print(' ',label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
ranks=F.rank(axis=1,pct=True); print('coverage',round(F.notna().mean().mean(),6),'mean_valid',round(F.notna().sum(axis=1).mean(),2),'turnover',round((ranks-ranks.shift()).abs().mean(axis=1).mean(),6),'cells',int(F.notna().sum().sum()),'cutoff',cut)
print('NOTE candidate panel generated for library audit; admitted factor signal panels not available in repository')
