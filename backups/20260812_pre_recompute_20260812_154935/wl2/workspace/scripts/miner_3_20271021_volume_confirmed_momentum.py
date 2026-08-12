import pandas as pd,numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2027-10-20'] for s in S}
p=pd.DataFrame({s:D[s].close for s in S}).sort_index(); vol=pd.DataFrame({s:D[s].volume for s in S}).sort_index(); r=p.pct_change()
# 5d trend confirmed by abnormal volume; volume is normalized to each instrument's 20d median.
abn=(vol/vol.rolling(20,min_periods=15).median()).replace([np.inf,-np.inf],np.nan).clip(.25,4)
f=(r.rolling(5).sum()*np.log(abn)).shift(1).ewm(span=3,min_periods=3,adjust=False).mean()
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],(p.shift(-1)/p-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']);q=o.ic
print('dates',len(o),'avgN',o.n.mean(),'coverage',o.n.sum()/(15*len(o)),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [3,5,10]:
 a=[]; y=p.shift(-h)/p-1
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=pd.Series(a);print('h',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'dates',len(a))
