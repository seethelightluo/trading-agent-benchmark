import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; P={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(P).sort_index().loc[:'2033-10-14']; rr=p.pct_change(); cs=rr.rolling(5).std().mean(axis=1); threshold=cs.rolling(60,min_periods=30).quantile(.65)
sig=(-rr.rolling(5).sum()).shift(1).mul((cs>threshold).shift(1),axis=0); sig=sig.sub(sig.mean(axis=1),axis=0)
for h in [1,3,5,10]:
 f=p.pct_change(h).shift(-h); a=[]; ns=[]
 for dt in sig.index:
  z=pd.DataFrame({'x':sig.loc[dt],'y':f.loc[dt]}).dropna()
  if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
   a.append(z.x.rank().corr(z.y.rank()));ns.append(len(z))
 a=np.array(a); print(h,len(a),round(np.mean(ns),2),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6),round((a>0).mean(),4))
print('active dates',int((sig.abs().sum(axis=1)>0).sum()),'span',p.index.min(),p.index.max())
sig.index=sig.index.strftime('%Y-%m-%d');sig.to_csv('scripts/miner_2_20331014_dispersion_reversal_signal.csv',index_label='date')
