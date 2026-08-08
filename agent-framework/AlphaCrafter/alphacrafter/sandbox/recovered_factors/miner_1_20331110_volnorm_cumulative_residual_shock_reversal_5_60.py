"""Miner 1: volatility-normalized cumulative idiosyncratic shock reversal, one idea."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2033-11-09')
p=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.rename(a) for a in A],axis=1).sort_index().loc[:END]
r=p.pct_change(fill_method=None); res=r.sub(r.median(axis=1),axis=0)
# Reversal after a persistent five-session, asset-specific displacement. Both the
# shock sum and normalizer end at t-1, so all inputs are completed at decision time.
shock=res.rolling(5,min_periods=5).sum().shift(1)
scale=res.rolling(60,min_periods=45).std().shift(1)*np.sqrt(5)
f=(-shock/scale).replace([np.inf,-np.inf],np.nan)
print('CANDIDATE volnorm_cumulative_residual_shock_reversal_5_60 endpoint',p.index.max().date(),'assets',len(A),'cells',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().mean().mean(),6),'eligible_dates',int(f.notna().sum(axis=1).ge(8).sum()))
R={}
for H in (1,5,10,20):
 fw=p.pct_change(H,fill_method=None).shift(-H); z=[]; ds=[]; nn=[]
 for t in f.index:
  q=pd.concat([f.loc[t],fw.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ds.append(t);nn.append(len(q))
 z=np.asarray(z);ds=pd.DatetimeIndex(ds); R[H]=(z,ds)
 print('H',H,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(nn),3),'PASS',bool(abs(z.mean())>=.007 and abs(z.mean()/z.std(ddof=1))>=.084))
best=max(R,key=lambda h:abs(R[h][0].mean()*R[h][0].mean()/R[h][0].std(ddof=1)); z,ds=R[best]
print('SELECTED_HORIZON',best)
for nm,a,b in [('2020_25','2020-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-31'),('2030_32','2030-01-01','2032-12-31'),('recent_2033','2033-01-01',END)]:
 x=z[(ds>=a)&(ds<=b)];print('REGIME',nm,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rnk=f.rank(axis=1,pct=True);tr=[]
for i in range(1,len(rnk)):
 q=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(q)>=8:tr.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('QUALITY turnover',round(np.mean(tr),6),'comparisons',len(tr),'median_iqr',round(f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median(),6))
