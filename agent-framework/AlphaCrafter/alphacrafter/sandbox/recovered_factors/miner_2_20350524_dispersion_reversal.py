import pandas as pd, numpy as np
from scipy.stats import rankdata
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for a in assets}).sort_index(); P=P[~P.index.duplicated(keep='last')]; P=P.loc[:'2035-05-23']; R=P.pct_change(); r5=P.pct_change(5)
disp=R.rolling(20,min_periods=12).std().mean(axis=1); act=(disp/disp.rolling(120,min_periods=40).median()).clip(.5,2.5)
F=-(r5.sub(r5.median(axis=1),axis=0)).mul(act.shift(1),axis=0); F=F.clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0).shift(1)
def run(h):
 z=[];ns=[];ds=[]
 fr=P.pct_change(h).shift(-h)
 for dt in F.index:
  x=np.column_stack((F.loc[dt].to_numpy(),fr.loc[dt].to_numpy())); ok=np.isfinite(x).all(1)
  if ok.sum()>=8:z.append(np.corrcoef(rankdata(x[ok,0]),rankdata(x[ok,1]))[0,1]);ns.append(ok.sum());ds.append(dt)
 z=np.array(z); return z,np.array(ns),pd.Index(ds)
print('cutoff',P.index.max(),'rows',len(P),'assets',P.shape[1],'cells',int(F.notna().sum().sum()),'coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 z,n,d=run(h);print('H',h,'IC %.6f ICIR %.6f dates %d meanN %.2f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),len(z),n.mean(),(z>0).mean()))
 for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2035-05-23')]:
  q=(d>=lo)&(d<=hi); zz=z[q]; print(' regime',lo,'n',len(zz),'ic %.6f icir %.6f'%(zz.mean(),zz.mean()/zz.std(ddof=1)) if len(zz)>1 else ' insufficient')
# correlation audit cannot reconstruct definitions; explicitly fail closed
print('max_abs_library_correlation unavailable; admission blocked')
