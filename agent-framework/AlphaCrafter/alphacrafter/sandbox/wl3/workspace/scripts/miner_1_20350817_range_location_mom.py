import pandas as pd, numpy as np, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cl={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); cl[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(cl).sort_index(); cutoff=pd.Timestamp('2035-08-16'); c=c[c.index<=cutoff]
r=c.pct_change()
# Range-location conditioned medium momentum: risk-adjusted 20d trend, strengthened
# when price is in the upper/lower half of its trailing 60d range (interpretable trend confirmation).
vol=r.rolling(60,min_periods=30).std()*np.sqrt(20)
trend=c.pct_change(20)/(vol+1e-12)
lo=c.rolling(60,min_periods=40).min(); hi=c.rolling(60,min_periods=40).max()
loc=(c-lo)/(hi-lo+1e-12)
f=trend*(0.5+loc)
rows=[]
for dt in f.index:
 x=f.loc[dt]; y=c.pct_change(1).shift(-1).loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].rank().corr(y[ok].rank()),ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=a.ic.dropna()
def ir(x): return x.mean()/x.std(ddof=1)*np.sqrt(len(x))
print('factor=range_location_conditioned_momentum20'); print('dates',len(a),'instruments',15,'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('1d_ic %.8f icir %.8f hit %.4f'%(q.mean(),ir(q),(q>0).mean()))
for h in [3,5,10,20]:
 y=c.pct_change(h).shift(-h); z=[]
 for dt in f.index:
  x=f.loc[dt]; yy=y.loc[dt]; ok=x.notna()&yy.notna()
  if ok.sum()>=8:z.append(x[ok].rank().corr(yy[ok].rank()))
 z=pd.Series(z).dropna(); print('horizon',h,'ic %.8f n %d'%(z.mean(),len(z)))
for name,x in [('early',q.iloc[:len(q)//3]),('middle',q.iloc[len(q)//3:2*len(q)//3]),('recent',q.iloc[2*len(q)//3:]),('recent120',q.tail(120))]: print(name,len(x),'ic %.8f icir %.8f hit %.4f'%(x.mean(),ir(x),(x>0).mean()))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',a.index.min().date(),a.index.max().date())
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20350817_range_location_mom_signal.csv',index=False)
