import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2028-12-31'); P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index();P[a]=d[d.index<=cut]
p=pd.DataFrame(P).sort_index();r=p.pct_change();ret=p/p.shift(20)-1;vol=r.rolling(30).std()*np.sqrt(252)
# relative reversal, defensive tilt in broad weakness, 25% tilt allocated as additive score
base=-ret/vol.replace(0,np.nan); breadth=(ret>0).mean(axis=1); defensive=['XAU','US10Y','CN10Y']; sig=base.copy()
for a in defensive: sig[a]=sig[a]+0.25*(breadth<.25).astype(float)
sig=sig.shift(1);f=p.shift(-10)/p-1; rows=[]
for d in p.index:
 x,y=sig.loc[d],f.loc[d];ok=x.notna()&y.notna()
 if ok.sum()>=8:rows.append((d,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');to=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).reindex(z.index)
print('candidate=defensive_relative_reversal025');print('dates',len(z),'avg_n',z.n.mean(),'period',z.index.min().date(),z.index.max().date());print('mean_ic %.6f icir %.6f hit %.4f turnover %.6f coverage %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean(),to.mean(),z.n.mean()/15))
for lo,hi in [('2025','2026'),('2027','2028')]:
 q=z.loc[lo:hi,'ic'];print(lo+'-'+hi,'n',len(q),'ic',q.mean(),'icir',q.mean()/q.std() if len(q)>1 else np.nan,'hit',(q>0).mean() if len(q) else np.nan)
sig.to_csv('scripts/miner_2_20290101_defensive025_signal.csv',index_label='date')
