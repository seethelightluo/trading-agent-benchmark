import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d['date'],errors='coerce'); D[s]=d.dropna(subset=['date']).sort_values('date').set_index('date')['close'].rename(s)
P=pd.concat(D.values(),axis=1,join='outer').sort_index(); ret=P.pct_change(); disp=ret.std(axis=1).rolling(5).mean(); z=(disp-disp.rolling(60).mean())/(disp.rolling(60).std()+1e-9)
for name,fac in [('conditional',-ret.shift(1)*(1+z.clip(-2,2).fillna(0).values[:,None])),('high_disp_only',-ret.shift(1)*z.clip(lower=0).fillna(0).values[:,None])]:
 for h in [1,5,10]:
  fw=P.shift(-h)/P-1; ics=[]; ns=[]
  for dt in P.index:
   g=pd.DataFrame({'f':fac.loc[dt],'r':fw.loc[dt]}).dropna()
   if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: ics.append(spearmanr(g.f,g.r).statistic);ns.append(len(g))
  a=np.array(ics); print(name,h,'dates',len(a),'avg_inst',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',len(ns)/len(P))
 zf=fac.rank(axis=1,pct=True); print('turn',zf.diff().abs().mean(axis=1).mean()*2)
