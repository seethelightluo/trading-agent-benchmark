import pandas as pd,numpy as np,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); c=c[c.index<=pd.Timestamp('2035-07-06')]; r=c.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').close.astype(float).reindex(c.index).ffill()
# Past-only volatility stress: VIX relative to its trailing 120-day mean, bounded attenuation.
zv=(v-v.rolling(120,min_periods=60).mean())/(v.rolling(120,min_periods=60).std()+1e-12)
stress=(1-0.20*zv).clip(0.55,1.25)
factor=c.pct_change(20)/(r.rolling(60,min_periods=40).std()+1e-12)*np.sign(c.pct_change(60))*stress.values[:,None]
y=c.pct_change(10).shift(-10)
rows=[]
for dt in factor.index:
 x=factor.loc[dt]; yy=y.loc[dt];ok=x.notna()&yy.notna()
 if ok.sum()>=8:
  rx=x[ok].rank(); ry=yy[ok].rank(); rows.append((dt,rx.corr(ry),ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic.dropna()
print('factor=vix_conditioned_confirmed_momentum20_vol60');print('dates',len(a),'instruments',15,'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('10d_ic %.8f icir %.8f hit %.4f'%(ic.mean(),ic.mean()/(ic.std(ddof=1)+1e-12)*np.sqrt(len(ic)),(ic>0).mean()))
for h in [1,5,20]:
 yy=c.pct_change(h).shift(-h);q=[]
 for dt in factor.index:
  x=factor.loc[dt]; z=yy.loc[dt];ok=x.notna()&z.notna()
  if ok.sum()>=8:q.append(x[ok].rank().corr(z[ok].rank()))
 print('horizon',h,'ic',np.nanmean(q),'n',len(q))
for name,x in [('early',ic.iloc[:len(ic)//3]),('middle',ic.iloc[len(ic)//3:2*len(ic)//3]),('recent',ic.iloc[2*len(ic)//3:]),('recent120',ic.tail(120))]: print(name,len(x),'ic',x.mean(),'icir',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',a.index.min().date(),a.index.max().date())
factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20350706_vix_confirm_signal.csv',index=False)
