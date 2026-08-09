import pandas as pd, numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2034-04-12')
def rd(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=E,'close'],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None); z=R['US10Y']-R['CN10Y']; w=30
# inverse transmission beta; all inputs lagged one day before scoring
vz=z.rolling(w,min_periods=20).var(); F=pd.DataFrame({a:-(R[a].rolling(w,min_periods=20).cov(z)/vz) for a in A}).shift(1)
print('factor inverse_rate_transmission_beta_30 visible',E.date(),'dates',len(P),'assets',len(A),'coverage',f'{F.notna().mean().mean():.6f}')
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; vals=[]; ns=[]
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: vals.append((t,q.f.corr(q.r,method='spearman'))); ns.append(len(q))
 x=pd.Series(dict(vals)); ics[h]=x; print('horizon',h,'dates',len(x),'mean_n',f'{np.mean(ns):.2f}','IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}','se',f'{x.std(ddof=1)/np.sqrt(len(x)):.6f}')
for lab,lo,hi in [('2020_24','2020','2024'),('2024_28','2024','2028'),('2028_31','2028','2031'),('2031_current','2031','2035')]:
 x=ics[20][(ics[20].index>=lo)&(ics[20].index<hi)]; print('regime_h20',lab,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True); tr=[]
for j in range(1,len(r)):
 q=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(q)>=8: tr.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('turnover',f'{np.mean(tr):.6f}','turnover_dates',len(tr)); print('library_correlation','NOT COMPUTED: exact common-cell audit required before admission')
