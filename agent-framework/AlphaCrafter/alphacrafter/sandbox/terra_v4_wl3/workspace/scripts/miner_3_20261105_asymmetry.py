import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close;P[s]=d
P=pd.DataFrame(P).sort_index().loc[:'2026-11-04'];R=P.pct_change(fill_method=None)
# asymmetry: upside average magnitude relative to downside magnitude, trailing 30 completed returns
up=R.clip(lower=0).rolling(30,min_periods=20).mean(); dn=(-R.clip(upper=0)).rolling(30,min_periods=20).mean(); F=(up-dn)/(up+dn).replace(0,np.nan)
for h in [1,5,10]:
 ic=[];ns=[]
 Y=P.pct_change(h,fill_method=None).shift(-h)
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],Y.iloc[i]],axis=1).dropna()
  if len(z)>=8:ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(ic);print(h,len(a),round(np.mean(ns),2),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0))
print('coverage',F.notna().sum().sum()/(len(F)*15),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
for y in range(2020,2027):
 a=[];Y=P.pct_change().shift(-1)
 for i in range(len(P)-1):
  if P.index[i].year==y:
   z=pd.concat([F.iloc[i],Y.iloc[i]],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(y,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1) if len(a)>1 else np.nan)
F.to_csv('scripts/miner_3_20261105_asymmetry_signal.csv',index_label='date')
