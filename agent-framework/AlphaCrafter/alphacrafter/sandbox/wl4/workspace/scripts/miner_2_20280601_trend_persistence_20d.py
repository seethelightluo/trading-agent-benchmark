import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; px={}
for a in A:
 f=f'{b}/{a}.csv'
 if os.path.exists(f): px[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close
P=pd.DataFrame(px).sort_index(); r=P.pct_change();
# persistence: signed fraction of up days, with magnitude confirmation via 20d return
f=((r>0).rolling(20,min_periods=15).mean()*2-1) * (P.pct_change(20).abs()+0.001)
f=f.shift(1)
for h in [1,5,10,20]:
 z=[];ns=[];ds=[]
 for d in f.index:
  x=f.loc[d]; y=(P.shift(-h)/P-1).loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   q=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(q): z.append(q); ns.append(ok.sum()); ds.append(d)
 s=pd.Series(z,index=ds); recent=s.tail(250)
 print('horizon',h,'dates',len(s),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/len(A),4),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent250',round(recent.mean(),6),round(recent.mean()/recent.std(ddof=1),6))
print('assets',len(px),'rows',len(P),'factor_coverage',round(f.notna().sum().sum()/(len(f)*len(px)),4))
