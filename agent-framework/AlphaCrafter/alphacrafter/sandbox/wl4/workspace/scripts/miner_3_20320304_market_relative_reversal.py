import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-02-04')
xs={}
for s in U:
 p=Path('../persistent/stock_data')/(s+'.csv')
 d=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].sort_index()
 xs[s]=d
px=pd.concat(xs,axis=1).sort_index().loc[:cut]
r=np.log(px).diff()
# relative 5d reversal: fade asset's recent return versus contemporaneous equal-weight market return
mkt=r.mean(axis=1)
f=( -(np.log(px).diff(5).sub(np.log(px).diff(5).mean(axis=1),axis=0)) ).shift(1)
rows=[]
for dt in f.index:
 if dt not in px.index: continue
 nxt=px.shift(-10).loc[dt]/px.loc[dt]-1
 z=pd.concat([f.loc[dt],nxt],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def stat(x): return x.ic.mean(), x.ic.mean()/x.ic.std(ddof=1), (x.ic>0).mean(),len(x),x.n.mean()
print('cut',cut.date(),'dates/n',len(a),a.n.mean())
for name,x in [('all',a),('365',a.loc[cut-pd.Timedelta(days=365):]),('730',a.loc[cut-pd.Timedelta(days=730):]),('1095',a.loc[cut-pd.Timedelta(days=1095):]),('2028-30',a.loc['2028':'2030']),('2020-22',a.loc['2020':'2022'])]: print(name,stat(x))
# rank turnover based on daily factor rankings
ranks=f.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).dropna().mean(); cov=f.notna().mean().mean()
print('coverage',cov,'turnover',turn)
for h in [5,10,20]:
 rr=px.shift(-h)/px-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('H',h,'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1), 'n',len(q))
