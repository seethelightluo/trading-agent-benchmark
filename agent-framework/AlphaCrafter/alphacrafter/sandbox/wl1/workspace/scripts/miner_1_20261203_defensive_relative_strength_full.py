import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-12-02'
def L(s):
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return x.close.loc[:cut]
p=pd.concat({s:L(s) for s in U},axis=1).sort_index();r=p.pct_change();E=U[:8]
eq=r[E].mean(axis=1); br=(r[E]>0).mean(axis=1)
st=((eq.rolling(10,min_periods=8).sum()<0)&(br.rolling(10,min_periods=8).mean()<.5)).astype(float)
rel=p.pct_change(20).sub(p.pct_change(20)[E].mean(axis=1),axis=0)
# defensive relative strength only when stress, neutral otherwise; signal lag
f=rel.mul(st.shift(1),axis=0).fillna(0).shift(1)
for h in [5,10,20]:
 fr=p.pct_change(h).shift(-h);a=[];ns=[];ds=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(d)
 a=np.array(a);print(h,len(a),np.mean(ns),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0))
 if h==10:
  for y in sorted(set(d.year for d in ds)):
   q=[v for d,v in zip(ds,a) if d.year==y];print(y,len(q),np.mean(q))
print('active',int(st.sum()),'coverage',f.notna().sum(axis=1).div(15).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
