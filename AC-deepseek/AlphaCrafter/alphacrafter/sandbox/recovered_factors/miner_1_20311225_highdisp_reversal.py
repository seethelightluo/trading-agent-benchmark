import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets}
px=pd.concat(D,axis=1).sort_index().ffill(); r=px.pct_change(); vol=r.rolling(20,min_periods=15).std()
med=r.median(axis=1); disp=r.sub(med,axis=0).abs().mean(axis=1); hi=disp.rolling(120,min_periods=60).rank(pct=True)>=.7
f=-(px.pct_change(5)/vol).where(hi.values[:,None]); fr=px.pct_change(1).shift(-1)
rows=[]
for dt in px.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('candidate high-disp volnorm reversal; dates',len(x),'meanN',x.n.mean(),'coverage',f.notna().stack().mean())
for a,b in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2031'),('2031-09-01','2031-12-25')]:
 q=x.loc[a:b].ic; print(a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
# turnover at 10d among signal ranks, conditional panels
rank=f.rank(axis=1,pct=True); print('turnover10d',rank.diff(10).abs().mean(axis=1).mean())
# correlation audit against broad reconstructed admitted proxies, candidate cells
proxies={'trend20':px.pct_change(20)/vol,'rev5':-px.pct_change(5)/r.rolling(5,min_periods=4).std(),'invvol':-vol,'trend60':px.pct_change(60),'skew40':-r.rolling(40,min_periods=30).skew(),'kurt40':-r.rolling(40,min_periods=30).kurt(),'gap3':-(pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').open/ pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.shift(1) for a in assets},axis=1).sort_index().ffill()-1).rolling(3).mean()}
mx=(0,None,0)
for k,v in proxies.items():
 z=pd.concat([f.stack().rename('c'),v.stack().rename('v')],axis=1).dropna()
 if len(z)>100:
  rho=spearmanr(z.c,z.v).statistic
  if abs(rho)>mx[0]: mx=(abs(rho),k,len(z))
print('max_abs_library_correlation',mx[0],'proxy',mx[1],'cells',mx[2], 'evidence=all finite reconstructed admitted-style signals screened')
print('decay')
for H in [1,5,10,20]:
 y=[]
 for dt in px.index:
  z=pd.concat([f.loc[dt],px.pct_change(H).shift(-H).loc[dt]],axis=1).dropna()
  if len(z)>=8:y.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(y);print(H,len(q),q.mean(),q.mean()/q.std(ddof=1))
