import pandas as pd,numpy as np,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data'
cl={}; vol={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'));d.date=pd.to_datetime(d.date);d=d.set_index('date');cl[s]=d.close.astype(float);vol[s]=d.volume.astype(float)
c=pd.DataFrame(cl).sort_index();v=pd.DataFrame(vol).reindex(c.index).sort_index();c=c[c.index<=pd.Timestamp('2035-07-20')];v=v.reindex(c.index)
r=c.pct_change()
# Volume-confirmed intermediate momentum: 10-session return scaled by relative recent activity.
vr=(v.rolling(20,min_periods=10).mean()/(v.rolling(60,min_periods=30).mean()+1e-12)).clip(0.5,2.0)
f=c.pct_change(10)*vr
yh={h:c.pct_change(h).shift(-h) for h in [1,5,10,20]}
rows=[]
for dt in f.index:
 x=f.loc[dt];y=yh[10].loc[dt];ok=x.notna()&y.notna()
 if ok.sum()>=8:rows.append((dt,x[ok].rank().corr(y[ok].rank()),ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');ic=a.ic.dropna()
print('factor=volume_confirmed_momentum10');print('dates',len(a),'instruments',15,'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('10d_ic %.8f icir %.8f hit %.4f'%(ic.mean(),ic.mean()/(ic.std(ddof=1)+1e-12)*np.sqrt(len(ic)),(ic>0).mean()))
for h,y in yh.items():
 q=[]
 for dt in f.index:
  x=f.loc[dt];z=y.loc[dt];ok=x.notna()&z.notna()
  if ok.sum()>=8:q.append(x[ok].rank().corr(z[ok].rank()))
 print('horizon',h,'ic',np.nanmean(q),'n',len(q))
for name,x in [('early',ic.iloc[:len(ic)//3]),('middle',ic.iloc[len(ic)//3:2*len(ic)//3]),('recent',ic.iloc[2*len(ic)//3:]),('recent120',ic.tail(120))]:print(name,len(x),'ic',x.mean(),'icir',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',a.index.min().date(),a.index.max().date())
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20350720_volume_confirm_signal.csv',index=False)
