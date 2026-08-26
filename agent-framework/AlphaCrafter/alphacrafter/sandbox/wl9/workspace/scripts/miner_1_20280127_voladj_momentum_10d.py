import os, numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    for root in ['../persistent/stock_data/','../persistent/index_data/']:
        f=root+s+'.csv'
        if os.path.exists(f):
            d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
            return d.close.astype(float)
    return None
series={s:load(s) for s in U}; series={s:v for s,v in series.items() if v is not None}
px=pd.DataFrame(series).sort_index().ffill(); ret=np.log(px).diff(); vol=ret.rolling(20,min_periods=15).std()
factor=ret.rolling(10,min_periods=10).sum()/vol; rows=[]
for dt in factor.index:
    z=pd.concat([factor.loc[dt],np.log(px.shift(-5)/px).loc[dt]],axis=1).dropna()
    if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
for name,q in [('all',r),('recent',r.loc[r.index>='2025-01-01']),('2026+',r.loc[r.index>='2026-01-01']),('2027+',r.loc[r.index>='2027-01-01'])]:
 if len(q): print(name,'dates',len(q),'meanIC',round(q.ic.mean(),5),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5),'hit',round((q.ic>0).mean(),3),'nmean',round(q.n.mean(),1))
ranks=factor.rank(axis=1,pct=True); print('coverage',round(factor.notna().sum(axis=1).mean()/len(U),3),'turnover',round(ranks.diff().abs().mean(axis=1).mean(),4),'decay')
for h in [1,3,5,10]:
 fw=np.log(px.shift(-h)/px); vals=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 vals=pd.Series(vals).dropna(); print(h,round(vals.mean(),5),round(vals.mean()/vals.std(ddof=1),5),len(vals))
print('assets',len(series),list(series))
