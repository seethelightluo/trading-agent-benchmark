import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); c=c[c.index<=pd.Timestamp('2035-07-05')]; r=c.pct_change()
# Volatility-compressed trend: risk-adjusted 20d momentum is trusted only when
# current 5d realized volatility is below its trailing 120d median.
m20=c.pct_change(20); vol60=r.rolling(60,min_periods=40).std(); v5=r.rolling(5,min_periods=5).std(); vmed=v5.rolling(120,min_periods=60).median()
factor=(m20/(vol60+1e-12))*((v5<=vmed).astype(float)*1.0 + (v5>vmed).astype(float)*0.35)
rows=[]
for dt in factor.index:
 y=c.pct_change(10).shift(-10).loc[dt]; x=factor.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=z.ic.dropna()
print('factor=compressed_trend20_vol60'); print('dates',len(z),'instruments',15,'avg_n',z.n.mean(),'coverage',z.n.mean()/15)
print('ic %.8f icir %.8f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),(a>0).mean()))
for h in [1,5,10,20]:
 q=[]
 for dt in factor.index:
  y=c.pct_change(h).shift(-h).loc[dt]; x=factor.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic)
 print('horizon',h,'ic',np.nanmean(q),'n',len(q))
for name,x in [('early',a.iloc[:len(a)//3]),('middle',a.iloc[len(a)//3:2*len(a)//3]),('recent',a.iloc[2*len(a)//3:]),('recent120',a.tail(120))]: print(name,len(x),'ic',x.mean(),'icir',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',z.index.min().date(),z.index.max().date())
sig=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();sig.to_csv('scripts/miner_2_20350706_compressed_trend_signal.csv',index=False)
