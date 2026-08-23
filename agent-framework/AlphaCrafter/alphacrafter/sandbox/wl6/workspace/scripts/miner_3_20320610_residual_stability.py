import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-06-09')
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close.astype(float) for s in U},axis=1).sort_index().loc[:cut]
r=P.pct_change(); b=r.mean(axis=1)
# beta-neutral residual returns; stability is inverse medium-term residual volatility,
# with a mild trend confirmation to avoid simply selecting stagnant assets.
var=b.rolling(60,min_periods=40).var(); beta=r.rolling(60,min_periods=40).cov(b).div(var,axis=0)
res=r-beta.mul(b,axis=0)
for w in [20,40]:
  rv=res.rolling(w,min_periods=max(10,w//2)).std()*np.sqrt(20)
  trend=res.rolling(20,min_periods=10).sum()
  f=-(rv/(1+trend.abs())) # stable residual behavior, interpretable
  print('variant',w,'coverage',f.notna().stack().mean())
  for h in [5,10,20]:
   fr=P.shift(-h)/P-1; vals=[];ns=[];ds=[]
   for dt in P.index:
    z=pd.concat([f.loc[dt].rename('x'),fr.loc[dt].rename('y')],axis=1).dropna()
    if len(z)>=8:
     q=z.x.corr(z.y,method='spearman')
     if np.isfinite(q): vals.append(q);ns.append(len(z));ds.append(dt)
   a=np.array(vals); q=pd.Series(a,index=ds)
   print('h',h,'valid_dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
   print('regimes',q.groupby(q.index.year).mean().round(5).to_dict())
