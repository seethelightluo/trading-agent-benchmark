import pandas as pd,numpy as np,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); c=c[c.index<=pd.Timestamp('2035-08-17')]; r=c.pct_change(); v=r.rolling(30,min_periods=20).std(); down=r.where(r<0,0).rolling(20,min_periods=15).std();
# Five-day reversal, risk-normalized; emphasize assets whose recent risk has been downside-skewed.
skew=(down/(v.rolling(20,min_periods=15).mean()+1e-12)).clip(.25,2.5)
f=(-c.pct_change(5)/(v+1e-12))*skew
y=c.pct_change(10).shift(-10); rows=[]
for dt in f.index:
 x=f.loc[dt]; z=y.loc[dt]; ok=x.notna()&z.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].rank().corr(z[ok].rank()),ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic.dropna(); print('factor downside_skew_scaled_reversal5'); print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15); print('10d_ic %.8f icir %.8f hit %.4f'%(ic.mean(),ic.mean()/(ic.std(ddof=1)+1e-12)*np.sqrt(len(ic)),(ic>0).mean()))
for h in [1,5,20]:
 z=c.pct_change(h).shift(-h);q=[]
 for dt in f.index:
  x=f.loc[dt]; zz=z.loc[dt];ok=x.notna()&zz.notna()
  if ok.sum()>=8:q.append(x[ok].rank().corr(zz[ok].rank()))
 print('horizon',h,'ic',np.nanmean(q),'n',len(q))
for name,x in [('early',ic.iloc[:len(ic)//3]),('middle',ic.iloc[len(ic)//3:2*len(ic)//3]),('recent',ic.iloc[2*len(ic)//3:]),('recent120',ic.tail(120))]:print(name,len(x),'ic',x.mean(),'icir',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',a.index.min().date(),a.index.max().date()); f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20350817_downside_skew_signal.csv',index=False)
