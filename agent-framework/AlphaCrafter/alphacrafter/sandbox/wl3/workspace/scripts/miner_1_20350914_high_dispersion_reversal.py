import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cl={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);cl[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(cl).sort_index();p=p[p.index<=pd.Timestamp('2035-09-13')];r=p.pct_change()
# Candidate: 5d reversal scaled by each asset's volatility, activated when cross-sectional dispersion is high.
rev=-p.pct_change(5); vol=r.rolling(30,min_periods=20).std()*np.sqrt(252); disp=r.std(axis=1).rolling(20,min_periods=15).mean(); threshold=disp.rolling(120,min_periods=60).median()
f=rev.div(vol.replace(0,np.nan)).mul((disp>threshold).astype(float),axis=0).replace([np.inf,-np.inf],np.nan)
rows=[]; y=p.shift(-10).div(p.shift(-1))-1
for dt in f.index:
 x=f.loc[dt];z=y.loc[dt];ok=x.notna()&z.notna()
 if ok.sum()>=8: rows.append((dt,x[ok].rank().corr(z[ok].rank()),ok.sum()))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=a.ic.dropna();print('factor=high_dispersion_volscaled_reversal5');print('dates',len(a),'instruments',15,'avg_n',a.n.mean(),'coverage',a.n.mean()/15);print('10d_ic %.8f icir %.8f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean()))
for h in [1,5,10,20]:
 yy=p.shift(-h).div(p.shift(-1))-1;zq=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:zq.append(f.loc[dt,ok].rank().corr(yy.loc[dt,ok].rank()))
 print('horizon',h,'ic',np.nanmean(zq),'n',len(zq))
for name,x in [('early',q.iloc[:len(q)//3]),('middle',q.iloc[len(q)//3:2*len(q)//3]),('recent',q.iloc[2*len(q)//3:]),('recent120',q.tail(120))]:print(name,len(x),'ic',x.mean(),'icir',x.mean()/x.std(ddof=1)*np.sqrt(len(x)),'hit',(x>0).mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',a.index.min().date(),a.index.max().date())
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20350914_high_dispersion_reversal_signal.csv',index=False)
