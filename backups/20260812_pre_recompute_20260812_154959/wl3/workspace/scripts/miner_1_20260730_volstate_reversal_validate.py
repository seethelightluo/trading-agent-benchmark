import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.split('/')[-1][:-4]
 if s in U:D[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15']
c=pd.DataFrame({s:d.close for s,d in D.items()}); r=c.pct_change(); v20=r.rolling(20,min_periods=10).std();v60=r.rolling(60,min_periods=20).std();x=-r*v60/v20; y=r.shift(-1)
out=[]; prev=None; turns=[]
for dt in x.index:
 z=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
  q=x.loc[dt].rank(pct=True).dropna(); turns.append(np.nan if prev is None else np.mean(abs(q-prev.reindex(q.index).fillna(.5))));prev=q
q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date');
for label,z in [('all',q),('early',q.iloc[:len(q)//2]),('late',q.iloc[len(q)//2:]),('recent250',q.iloc[-250:])]:
 ic=z.ic;print(label,'dates',len(z),'avg_n',z.n.mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1)*np.sqrt(252),'hit',np.mean(ic>0))
print('turnover',np.nanmean(turns),'coverage',q.n.sum()/(len(q)*15))
