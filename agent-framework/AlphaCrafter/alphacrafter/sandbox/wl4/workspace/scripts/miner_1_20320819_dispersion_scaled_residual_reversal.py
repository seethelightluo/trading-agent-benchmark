import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2032-08-19')
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']; D[s]=x[x.index<=CUT]
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); bench=r.mean(axis=1); resid=r.sub(bench,axis=0)
resid20=resid.rolling(20,min_periods=15).sum(); down60=r.clip(upper=0).pow(2).rolling(60,min_periods=40).mean().pow(.5)*np.sqrt(252)
disp20=r.std(axis=1).rolling(20,min_periods=15).mean(); ratio=(disp20/disp20.rolling(120,min_periods=60).median()).clip(0.5,2.0)
sig=(-(resid20/down60.replace(0,np.nan)).mul(ratio,axis=0)).shift(1)
for h in [10,20,30]:
 f=p.shift(-h)/p-1; vals=[]; ns=[]; turns=[]; prev=None
 for d in sig.index:
  q=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(q)>=8:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q)); rk=q.iloc[:,0].rank(pct=True)
   if prev is not None: turns.append(np.mean(abs(rk-prev)))
   prev=rk
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4),'turnover',round(np.nanmean(turns),6))
h=20; f=p.shift(-h)/p-1
for label,mask in [('2020-27',sig.index<'2028-01-01'),('2028-30',(sig.index>='2028-01-01')&(sig.index<'2031-01-01')),('2031-32',sig.index>='2031-01-01'),('recent365',sig.index>=CUT-pd.Timedelta(days=365))]:
 a=[]
 for d in sig.index[mask]:
  q=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 a=np.array(a);print(label,'dates',len(a),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),6))
print('last',p.index.max().date())
