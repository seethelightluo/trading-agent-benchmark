import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date') for s in U}
p=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index();o=pd.DataFrame({s:d.open for s,d in D.items()}).reindex(p.index);h=pd.DataFrame({s:d.high for s,d in D.items()}).reindex(p.index);l=pd.DataFrame({s:d.low for s,d in D.items()}).reindex(p.index);r=p.pct_change();
# normalized intraday reversal: bearish close relative to open/range predicts reversal; higher means defensive reversal
f=-(p-o)/(h-l).replace(0,np.nan)
ic=[];ns=[];ds=[];rr=[]
for i in range(len(p)-1):
 q=pd.concat([f.iloc[i],r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(q)>=8:ic.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q));ds.append(p.index[i]);rr.append(q.iloc[:,0].rank(pct=True))
x=np.array(ic);print('dates',len(x),'N',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'turn',round(np.nanmean([np.abs(rr[j]-rr[j-1]).mean() for j in range(1,len(rr))]),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 z=x[(pd.DatetimeIndex(ds).year>=lo)&(pd.DatetimeIndex(ds).year<=hi)];print('regime',lo,hi,'IC',round(z.mean(),6),'n',len(z))
clv=-(2*(p-l)/(h-l).replace(0,np.nan)-1);print('corr_clv',f.stack().corr(clv.stack()))
