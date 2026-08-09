"""One candidate: residual-downside overnight-gap repair acceleration, cutoff 2032-04-14."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-04-14'); START=pd.Timestamp('2026-07-16')
def load(a,col):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[START:END,col].astype(float)
c=pd.DataFrame({a:load(a,'close') for a in A}); o=pd.DataFrame({a:load(a,'open') for a in A})
r=c.pct_change(fill_method=None); m=r.median(axis=1)
b=r.rolling(60,min_periods=45).cov(m).div(m.rolling(60,min_periods=45).var(),axis=0); resid=r-b.mul(m,axis=0)
# Following an idiosyncratic down day, a positive next-session open-to-close return
# denotes repair. Contrast its recent frequency-weighted magnitude with its 60d norm.
intra=c.div(o).sub(1); scale=intra.rolling(60,min_periods=45).std()
event=resid.shift(1)<-resid.rolling(60,min_periods=45).std().shift(1)
repair=(intra/scale).clip(-5,5).where(event)
f=repair.rolling(20,min_periods=6).mean()-repair.rolling(60,min_periods=15).mean()
print('FACTOR residual_downside_gap_repair_acceleration_20_60obs cutoff',END.date(),'assets',len(A))
print('construction: 20d minus 60d mean volatility-normalized open-to-close return conditional on previous residual downside shock')
print('coverage',int(f.notna().sum().sum()),'/',f.size,round(f.notna().mean().mean(),6))
R={}
for h in [1,5,10,20]:
 y=c.shift(-h).div(c).sub(1); z=[]; dates=[]; ns=[]
 for t in f.index:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);dates.append(t);ns.append(len(q))
 z=np.array(z);dates=pd.DatetimeIndex(dates); R[h]=(z,dates)
 print('H',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6),'mean_n',round(np.mean(ns),3),'min_n',min(ns))
for name,lo in [('2026_2029','2026-07-16'),('2030_2032','2030-01-01'),('recent_12m','2031-04-15')]:
 z,d=R[20]; x=z[d>=lo];print('REGIME',name,'H20 dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rnk=f.rank(axis=1,pct=True); dif=(rnk-rnk.shift()).abs(); print('TURNOVER daily_rank_abs_change',round(dif.stack().mean(),6),'median_iqr',round((f.quantile(.75,axis=1)-f.quantile(.25,axis=1)).median(),6))
print('NOVELTY: not computed. This preliminary candidate is not eligible for admission without an exact full-library signal correlation audit.')
