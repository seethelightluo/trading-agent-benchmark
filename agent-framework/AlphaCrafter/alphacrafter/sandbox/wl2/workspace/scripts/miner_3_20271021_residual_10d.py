import pandas as pd,numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:'2027-10-20'] for s in S}
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); v=r.rolling(20,min_periods=15).std()
# 10-day risk-adjusted return residualized cross-sectionally to the contemporaneous median.
base=(p.pct_change(10)/v).replace([np.inf,-np.inf],np.nan)
f=base.sub(base.median(axis=1),axis=0).shift(1).ewm(span=3,min_periods=3,adjust=False).mean()
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=o.ic
print('dates',len(o),'avgN',o.n.mean(),'coverage',o.n.sum()/(len(o)*15),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [3,5,10]:
 yy=p.shift(-h)/p-1;a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=pd.Series(a);print('h',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'dates',len(a))
# regimes by median 20d cross-sectional return
reg=(p.pct_change(20).median(axis=1)>0)
for name,mask in [('up',reg),('down',~reg)]:
 z=q.reindex(q.index.intersection(reg.index))[mask.reindex(q.index).fillna(False).values if name=='up' else (~reg).reindex(q.index).fillna(False).values]
 print(name,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
