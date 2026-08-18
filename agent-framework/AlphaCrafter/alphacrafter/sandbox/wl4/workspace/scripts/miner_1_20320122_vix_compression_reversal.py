import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-01-21')
p={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); p[s]=d.close[d.index<=cut]
px=pd.DataFrame(p).sort_index().ffill(); r=px.pct_change(); R=px.pct_change(10); res=R.sub(R.median(axis=1),axis=0); vol=r.rolling(40).std();
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(px.index).ffill(); vp=vix.rolling(120,min_periods=60).rank(pct=True)
# mean-reversion signal activated/weighted in calm-to-normal volatility, strictly lagged
f=(-res/vol * (1.25-0.5*vp)).shift(1); fr=px.shift(-10)/px-1
vals=[];ds=[];ns=[]; prev=None; turns=[]
for t in f.index:
 z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(t);ns.append(len(z))
  q=f.loc[t].rank(pct=True); turns.append(np.nan if prev is None else (q-prev).abs().mean());prev=q
x=pd.Series(vals,index=ds); print('factor vix_compression_reversal_10d');print('dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'coverage',round(len(x)/len(f),4),'turnover',round(np.nanmean(turns),4),'end',x.index.max().date())
for n in [365,730,1095]:
 y=x[x.index>=x.index.max()-pd.Timedelta(days=n)];print('recent',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),4))
for h in [5,10,20]:
 ff=px.shift(-h)/px-1; a=[]
 for t in f.index:
  z=pd.concat([f.loc[t],ff.loc[t]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print('decay',h,'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6))
