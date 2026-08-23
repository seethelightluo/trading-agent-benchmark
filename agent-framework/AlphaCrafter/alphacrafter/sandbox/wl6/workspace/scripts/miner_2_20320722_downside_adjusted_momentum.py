import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-07-21'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:cut]
 r=x.close.pct_change(); down=r.where(r<0,0.0).rolling(20,min_periods=15).std()
 x['f']=x.close.pct_change(40)/(down*np.sqrt(252)+1e-12)
 for h in (5,10,20): x[f'r{h}']=x.close.shift(-h)/x.close-1
 D[s]=x[['f','r5','r10','r20']]
dates=sorted(set.intersection(*[set(x.index) for x in D.values()]))
for h in ['r5','r10','r20']:
 vals=[]; used=[]; ns=[]; ranks=[]
 for dt in dates:
  z=pd.DataFrame({s:[D[s].loc[dt,'f'],D[s].loc[dt,h]] for s in U},index=['f','r']).T.dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.f,z.r).statistic); used.append(dt); ns.append(len(z))
   ranks.append(pd.Series(z.f.rank(pct=True),index=z.index).reindex(U).fillna(.5).values)
 a=np.array(vals); print(h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(float(np.nanmean(a)),6),'ICIR_daily',round(float(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12)),6),'ICIR_ann',round(float(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12)*np.sqrt(252)),6),'hit',round(float(np.mean(a>0)),4),'cov',round(float(np.mean(ns)/15),4),'turnover',round(float(np.nanmean(np.abs(np.diff(np.array(ranks),axis=0)))),5))
 for y in range(2026,2033):
  q=a[[d.year==y for d in used]]; print(' ',y,round(float(np.nanmean(q)),5) if len(q) else None,len(q))
