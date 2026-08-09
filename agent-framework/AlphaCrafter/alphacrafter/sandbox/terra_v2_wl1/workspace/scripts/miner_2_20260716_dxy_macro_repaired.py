import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root='../persistent'
def load(path):
 d=pd.read_csv(path,parse_dates=['date']).set_index('date').sort_index(); return d.close.astype(float)
px=pd.concat({s:load(root+'/stock_data/'+s+'.csv') for s in U},axis=1).sort_index().loc[:'2026-07-15']
# Keep each asset's observations on its own calendar; DXY is aligned only after level forward-fill.
dxy=load(root+'/index_data/DXY.csv').reindex(px.index).ffill(); r=px.pct_change(fill_method=None); dr=dxy.pct_change(fill_method=None)
beta=pd.DataFrame(index=px.index,columns=U,dtype=float); residual=pd.DataFrame(index=px.index,columns=U,dtype=float)
for s in U:
 z=pd.concat([r[s],dr],axis=1).dropna(); z.columns=['x','m']
 b=z.x.rolling(60,min_periods=40).cov(z.m)/z.m.rolling(60,min_periods=40).var()
 beta[s]=b.reindex(px.index)
 residual[s]=(z.x-b*z.m).rolling(10,min_periods=8).sum().reindex(px.index)
fw=px.shift(-1)/px-1
def evaluate(name,fac,fwd):
 vals=[]; ns=[]; ds=[]
 for dt in fac.index:
  q=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v); ns.append(len(q)); ds.append(dt)
 a=pd.Series(vals,index=pd.to_datetime(ds)); ic=a.mean(); ir=ic/a.std(ddof=1)
 turn=fac.rank(axis=1,pct=True).diff().abs().mean().mean()
 print(name,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sum(ns)/(len(a)*15),4),'IC',round(ic,5),'ICIR',round(ir,5),'hit',round((a>0).mean(),4),'turn',round(turn,4),'annual',a.groupby(a.index.year).mean().round(4).to_dict())
evaluate('dxy_beta_negative_60d',-beta,fw); evaluate('dxy_residual_momentum_10obs',residual,fw)
for h in [5,10]:
 fwd=px.shift(-h)/px-1; evaluate('beta_neg_h'+str(h),-beta,fwd); evaluate('residual_h'+str(h),residual,fwd)
print('valid beta',int(beta.notna().sum().sum()),'residual',int(residual.notna().sum().sum()))
