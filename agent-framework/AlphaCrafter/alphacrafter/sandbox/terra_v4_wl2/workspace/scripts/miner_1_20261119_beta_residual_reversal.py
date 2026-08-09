import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(p).sort_index(); r=p.pct_change(); m=r.median(axis=1)
beta=pd.DataFrame({s:r[s].rolling(60,min_periods=40).cov(m)/m.rolling(60,min_periods=40).var() for s in U},index=p.index)
for L in [3,5,10]:
 f=-(r.rolling(L).sum()-beta.mul(m.rolling(L).sum(),axis=0))
 for h in [1,5,10]:
  qs=[]; ns=[]; ranks=[]; yrs=[]
  for i in range(65,len(p)-h):
   z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1:
    qs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); ranks.append(z.iloc[:,0].rank(pct=True)); yrs.append(p.index[i].year)
  q=np.array(qs); turn=np.nanmean([np.mean(np.abs(ranks[j]-ranks[j-1])) for j in range(1,len(ranks))])
  print('L',L,'h',h,'dates',len(q),'avg_names',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4),'turn',round(turn,4),'years',min(yrs),max(yrs))
  for a,b in [(2020,2022),(2023,2024),(2025,2026)]:
   z=q[(np.array(yrs)>=a)&(np.array(yrs)<=b)]
   print(' regime',a,b,'n',len(z),'ICIR',round(z.mean()/z.std(ddof=1),4) if len(z)>1 else None)
print('dates',len(p),'instruments',len(U),'range',p.index.min(),p.index.max())
