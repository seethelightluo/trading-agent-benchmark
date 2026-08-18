import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-03-02')
px={}
for a in assets:
    f='../persistent/stock_data/'+a+'.csv'; d=pd.read_csv(f); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').close
    px[a]=d
p=pd.DataFrame(px).sort_index().ffill()
r=np.log(p).diff()
# one interpretable idea: lagged residual shock reversal, compression gated
# residual = asset 5d return less cross-sectional median; inverse sign, vol-normalized, weighted by low vol state
ret5=r.rolling(5).sum(); med=ret5.median(axis=1); resid=ret5.sub(med,axis=0)
vol20=r.rolling(20).std()*np.sqrt(252)
shock=(-resid/vol20).clip(-4,4)
comp=(r.rolling(10).std()/r.rolling(60).std()).clip(0.25,2.0)
gate=(1-comp).clip(-1,1)
# positive gate means compression; preserve reversal direction, shrink in expansion
sig=shock*(1+gate).shift(1)
rows=[]
for h in [10,20,40]:
    fwd=np.log(p.shift(-h)/p)
    ics=[]
    for dt in sig.index:
        x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    q=pd.Series(ics).dropna(); print(h,'dates',len(q),'avgN',15,'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(),(q>0).mean()))
# regime on 20d
h=20; fwd=np.log(p.shift(-h)/p); ics=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(ics,columns=['date','ic']);q.date=pd.to_datetime(q.date)
for label,mask in [('2026-28',(q.date>='2026')&(q.date<'2029')),('2029-33',(q.date>='2029')&(q.date<'2034')),('recent',(q.date>='2031'))]:
 x=q.loc[mask,'ic']; print(label,len(x),x.mean(),x.mean()/x.std(),(x>0).mean())
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
# artifact for provenance
out=[]
for dt in sig.index:
 for a in assets:
  if pd.notna(sig.loc[dt,a]): out.append([dt,a,sig.loc[dt,a],fwd.loc[dt,a]])
pd.DataFrame(out,columns=['date','asset','signal','fwd20']).to_csv('scripts/miner_1_20340303_residual_shock_reversal_20d_signal.csv',index=False)
q.to_csv('scripts/miner_1_20340303_residual_shock_reversal_20d_ic.csv',index=False)
