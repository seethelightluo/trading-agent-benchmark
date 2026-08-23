import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-07-21'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:cut]; r=x.close.pct_change(); d=r.where(r<0,0).rolling(20,min_periods=15).std(); x['f']=x.close.pct_change(40)/(d*np.sqrt(252)+1e-12)
 for h in (5,10,20): x[f'r{h}']=x.close.shift(-h)/x.close-1
 D[s]=x[['f','r5','r10','r20']]
dates=sorted(set.intersection(*[set(x.index) for x in D.values()]))
for h in ['r5','r10','r20']:
 a=[]; ns=[]; ranks=[]; used=[]
 for dt in dates:
  z=pd.DataFrame({s:{'f':D[s].loc[dt,'f'],'r':D[s].loc[dt,h]} for s in U}).T.dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:
   a.append(spearmanr(z.f,z.r).statistic);ns.append(len(z));used.append(dt);ranks.append(z.f.rank(pct=True).reindex(U).fillna(.5).values)
 a=np.array(a); sd=np.nanstd(a,ddof=1); print(h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR_daily',round(np.nanmean(a)/(sd+1e-12),6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4),'turnover',round(np.nanmean(np.abs(np.diff(np.array(ranks),axis=0))),5))
 print('regimes', {y:round(float(np.nanmean(a[[d.year==y for d in used]])),5) for y in range(2026,2033) if any(d.year==y for d in used)})
