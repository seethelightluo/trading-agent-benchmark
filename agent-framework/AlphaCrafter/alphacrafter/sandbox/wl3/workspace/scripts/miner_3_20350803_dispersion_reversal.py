import pandas as pd,numpy as np,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];base='../persistent/stock_data'
cl={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'));d.date=pd.to_datetime(d.date);cl[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(cl).sort_index(); c=c[c.index<=pd.Timestamp('2035-08-03')]
r=c.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Dispersion-conditioned short reversal: reverse each asset's 5d move, amplified only
# when cross-asset dispersion is elevated; all inputs are observable at signal date.
disp=r.rolling(20,min_periods=15).std().mean(axis=1)
med=disp.rolling(60,min_periods=30).median(); gate=(disp/(med+1e-12)).clip(.5,2.0)
f=(-c.pct_change(5)/(vol*np.sqrt(5)+1e-12))*gate.values[:,None]
y=c.pct_change(10).shift(-10)
rows=[]
for dt in f.index:
 x=f.loc[dt]; z=y.loc[dt]; ok=x.notna()&z.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].rank().corr(z[ok].rank()),ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=a.ic.dropna()
print('factor=dispersion_conditioned_reversal5'); print('dates',len(a),'instruments',15,'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('10d_ic %.8f icir %.8f hit %.4f'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(len(q)),(q>0).mean()))
for h in [1,5,10,20]:
 yy=c.pct_change(h).shift(-h); zq=[]
 for dt in f.index:
  x=f.loc[dt]; z=yy.loc[dt];ok=x.notna()&z.notna()
  if ok.sum()>=8:zq.append(x[ok].rank().corr(z[ok].rank()))
 print('horizon',h,'ic',np.nanmean(zq),'n',len(zq))
for name,x in [('early',q.iloc[:len(q)//3]),('middle',q.iloc[len(q)//3:2*len(q)//3]),('recent',q.iloc[2*len(q)//3:]),('recent120',q.tail(120))]: print(name,len(x),'ic',x.mean(),'icir',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',a.index.min().date(),a.index.max().date())
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20350803_dispersion_reversal_signal.csv',index=False)
