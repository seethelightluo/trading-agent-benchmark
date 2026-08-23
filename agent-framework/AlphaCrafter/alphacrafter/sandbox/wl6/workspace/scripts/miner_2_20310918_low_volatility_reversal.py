import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); px[s]=d.close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:'2031-09-17']; R=P.pct_change()
# Lower trailing realized volatility receives higher score; cross-sectional rank handles scale.
f=-R.rolling(20,min_periods=15).std()
print('candidate=low_volatility_reversal; dates=',len(P),'instruments=',len(P.columns),'through=',P.index.max().date())
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; out=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): out.append((dt,q,len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('H',h,'valid_dates',len(q),'avg_n',round(q.n.mean(),3),'IC',round(ic,8),'ICIR_daily',round(ic/sd,8),'ICIR_sqrtN',round(ic/sd*np.sqrt(len(q)),8),'hit',round((q.ic>0).mean(),4),'years',q.groupby(q.index.year).ic.mean().round(5).to_dict())
print('coverage',round(f.notna().sum(axis=1).mean()/15,6),'turnover_proxy',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
