import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-16'); P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float); P[s]=d[d.index<=cut]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); neg2=(r.clip(upper=0)**2)
# downside deviation: square root of mean squared negative returns, with all days in window
for w in [20,40,60]:
 dd=np.sqrt(neg2.rolling(w,min_periods=w//2).mean())*np.sqrt(252)
 F=P.pct_change(20).div(dd.replace(0,np.nan)); print('WINDOW',w)
 for h in [1,5,10]:
  Y=P.shift(-h).div(P)-1; vals=[]
  for dt in P.index:
   z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
   if len(z)>=8: vals.append(z.f.corr(z.y,method='spearman'))
  ic=pd.Series(vals).dropna(); print('H',h,'dates',len(ic),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 print('coverage',round(F.notna().sum().sum()/F.size,4))
