import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U}
p=pd.concat(D,axis=1,sort=True).loc[:'2026-07-15']; r=np.log(p).diff(); x=r['SPX']; wx=x.rolling(60,min_periods=40); mx=wx.mean(); vx=wx.var(); b=pd.DataFrame(index=r.index,columns=U,dtype=float)
for s in U:
 mr=r[s].rolling(60,min_periods=40).mean(); mrx=(r[s]*x).rolling(60,min_periods=40).mean(); b[s]=(mrx-mr*mx)/vx
res=r.rolling(5,min_periods=5).sum()-b.mul(x.rolling(5,min_periods=5).sum(),axis=0); f=-res
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt], 'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r,method='spearman'));ns.append(len(a));dates.append(dt)
 z=np.array(vals); dt=np.array(dates,dtype='datetime64[ns]'); print('h',h,'dates',len(z),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 if h==1:
  print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
  for label,a in [('2020-22',z[dt<='2022-12-31']),('2023-24',z[(dt>='2023-01-01')&(dt<='2024-12-31')]),('2025-26',z[dt>='2025-01-01'])]: print(label,len(a),a.mean(),a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
print('period',p.index.min(),p.index.max())
