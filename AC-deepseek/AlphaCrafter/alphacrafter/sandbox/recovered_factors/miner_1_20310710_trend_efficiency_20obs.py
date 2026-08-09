import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); px[a]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Trend efficiency: directional displacement divided by total path, smoothed 20d;
# signed to distinguish persistent up/down paths, with volatility normalization.
path=r.abs().rolling(20,min_periods=15).sum(); disp=p.pct_change(20); f=(disp/path.replace(0,np.nan)).clip(-2,2)

def calc(h):
 vals=[]; ns=[]; dates=[]
 for i in range(len(p)-h):
  x=f.iloc[i]; y=p.iloc[i+h]/p.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum()); dates.append(p.index[i])
 s=pd.Series(vals,index=dates); return s,np.mean(ns)
for h in [1,5,10,20]:
 s,mn=calc(h); print(f'H{h}: dates={len(s)} meanN={mn:.2f} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1):.6f} hit={np.mean(s>0):.3f}')
s,mn=calc(10)
for label,mask in [('2020-23',s.index<'2024-01-01'),('2024-27',(s.index>='2024-01-01')&(s.index<'2028-01-01')),('2028-30',(s.index>='2028-01-01')&(s.index<'2031-01-01')),('2031',s.index>='2031-01-01'),('latest120',s.index>=s.index[-120])]:
 z=s[mask]; print(label,len(z),f'IC={z.mean():.6f}',f'ICIR={z.mean()/z.std(ddof=1):.6f}')
print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]))
print('turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).dropna().mean())
for name,q in [('mom20',p.pct_change(20)),('volnorm',p.pct_change(20)/r.rolling(20,min_periods=15).std()),('path',path)]:
 z=pd.concat([f.stack(),q.stack()],axis=1).dropna(); print('corr',name,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z))
