import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']) for s in U}
P=pd.concat({s:d.set_index('date')['close'] for s,d in D.items()},axis=1).sort_index(); R=np.log(P).diff()
r5=R.rolling(5).sum(); r10=R.rolling(10).sum(); vol=R.rolling(40).std()*np.sqrt(252)
# Cross-sectional residual of 10d return after removing contemporaneous 5d return and market/common component.
res=pd.DataFrame(index=R.index,columns=U,dtype=float)
for dt in R.index:
 x=r5.loc[dt]; y=r10.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  xx=x[ok].values; yy=y[ok].values
  A=np.column_stack([np.ones(len(xx)),xx])
  b=np.linalg.lstsq(A,yy,rcond=None)[0]
  res.loc[dt,ok]=yy-A@b
f=(-res/(vol+1e-8)).clip(-8,8).shift(1)
# same horizon 10d forward, plus 1d and decay
fw={n:R.shift(-1).rolling(n).sum().shift(-(n-1)) for n in [1,3,5,10,20]}
ics={}; ns={}
for n,y in fw.items():
 vals=[]; dates=[]; counts=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   z=f.loc[dt,ok].corr(y.loc[dt,ok],method='spearman')
   if pd.notna(z): vals.append(z); dates.append(dt); counts.append(ok.sum())
 s=pd.Series(vals,index=dates); ics[n]=s
 print('horizon',n,'dates',len(s),'avg_n',round(np.mean(counts),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().sum(axis=1)/f.notna().sum(axis=1)).mean(),6))
s=ics[10]
for label,ix in [('early',s.index<='2023-08-31'),('middle',(s.index>'2023-08-31')&(s.index<='2027-05-31')),('late',s.index>'2027-05-31')]: print(label,'IC',round(s[ix].mean(),6),'dates',int(ix.sum()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20301230_residual_reversal10_signal.csv',index=False)
