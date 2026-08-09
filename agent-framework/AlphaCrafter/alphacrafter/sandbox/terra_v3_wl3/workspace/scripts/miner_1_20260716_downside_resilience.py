import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:'2026-07-15']; p.columns=U; r=np.log(p).diff(); m=r.median(axis=1)
th=m.rolling(120,min_periods=60).quantile(.20); stress=(m<=th).fillna(False); st=stress.astype(float)
sm=r.mul(st,axis=0).rolling(60,min_periods=30).sum().div(st.rolling(60,min_periods=30).sum(),axis=0)
nm=r.mul(1-st,axis=0).rolling(60,min_periods=30).sum().div((1-st).rolling(60,min_periods=30).sum(),axis=0); f=sm-nm
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); z=[]; ns=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1:z.append(a.f.corr(a.r,method='spearman'));ns.append(len(a))
 z=np.array(z);print('h',h,'dates',len(z),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
q=f.rank(axis=1,pct=True);print('turnover',q.diff().abs().mean(axis=1).mean(),'coverage',f.notna().sum().sum()/f.size,'stressdays',stress.sum())
print('early/late/recent',[(a.mean(),a.mean()/a.std(ddof=1),len(a)) for a in [z[:len(z)//2],z[len(z)//2:],z[-250:]]])
print('corr rev5',f.stack().corr((-p.pct_change(5)).stack()))
