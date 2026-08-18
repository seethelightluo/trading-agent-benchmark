import numpy as np,pandas as pd,glob
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-08-06'); D={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in U:D[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:cut]
px=pd.concat({s:D[s].close for s in U if s in D},axis=1).sort_index();r=px.pct_change();vol=r.rolling(20,min_periods=15).std();down=(r.clip(upper=0)**2).rolling(20,min_periods=15).mean().pow(.5);up=(r.clip(lower=0)**2).rolling(20,min_periods=15).mean().pow(.5)
f=(1/vol/(1+down/(up+1e-8))).shift(1)
for h in [5,10,20]:
 y=px.pct_change(h).shift(-h);a=[];rank=[];ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   v=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(v):a.append(v);rank.append(f.iloc[i].rank(pct=True));ns.append(len(z))
 a=np.array(a);print('H%d dates=%d avgN=%.2f cov=%.4f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.4f'%(h,len(a),np.mean(ns),np.mean(ns)/15,a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.mean([(rank[i]-rank[i-1]).abs().mean() for i in range(1,len(rank))])))
 for w in [365,730,1095]:
  q=a[-w:];print('recent%d IC=%.6f ICIR=%.6f'%(w,q.mean(),q.mean()/q.std(ddof=1)))
