import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for root in ['../persistent/stock_data/','../persistent/index_data/']:
  f=root+s+'.csv'
  if os.path.exists(f): return pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close.astype(float)
D={s:load(s) for s in U}; D={s:v for s,v in D.items() if v is not None}; P=pd.DataFrame(D).sort_index().ffill(); R=np.log(P).diff();
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.reindex(P.index).ffill(); vz=(v>v.rolling(60,min_periods=40).median()).astype(float)
# reversal only when VIX is elevated, zero otherwise; factor ranks remain cross-sectional due to raw reversal
F=-R.rolling(5,min_periods=5).sum().mul(vz,axis=0); FW=np.log(P.shift(-5)/P); rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],FW.loc[dt]],axis=1).dropna()
 if len(z)>=8 and vz.loc[dt]>0: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
for name,q in [('all',r),('2025+',r.loc['2025-01-01':]),('2026+',r.loc['2026-01-01':]),('2027+',r.loc['2027-01-01':])]:
 if len(q): print(name,'dates',len(q),'meanIC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5),'hit',round((q.ic>0).mean(),3),'nmean',round(q.n.mean(),1))
print('coverage_active',round((F.notna().sum(axis=1)[vz>0].mean()/len(U)),3),'active_fraction',round(vz.mean(),3),'assets',len(D))
for h in [1,3,5,10]:
 fw=np.log(P.shift(-h)/P); a=[]
 for dt in F.index:
  if vz.loc[dt]>0:
   z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a).dropna(); print('decay',h,round(a.mean(),5),round(a.mean()/a.std(ddof=1),5),len(a))
