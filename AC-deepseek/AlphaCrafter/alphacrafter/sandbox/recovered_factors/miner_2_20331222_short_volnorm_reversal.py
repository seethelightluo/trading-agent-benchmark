import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index().close
p=pd.DataFrame({a:ld(a) for a in A}); ix=p.index; r=p.pct_change(); vol=r.rolling(20).std()*np.sqrt(20)
# Short-horizon reversal normalized by current risk, lagged one day.
sig=(-p.pct_change(3)/(vol+1e-8)).shift(1)
print('through',ix[-1].date(),'dates',len(ix),'assets',len(A))
for h in [1,5,10,20]:
 f=p.pct_change(h).shift(-h); z=[]; n=[]
 for d in ix:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic);n.append(ok.sum())
 z=np.array(z);print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(z.mean(),z.mean()/(z.std(ddof=1)+1e-12),(z>0).mean(),len(z),np.mean(n)))
f=p.pct_change(10).shift(-10); rows=[]
for d in ix:
 ok=sig.loc[d].notna()&f.loc[d].notna()
 if ok.sum()>=8:rows.append((d,spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic))
for lo,hi in [(2020,2023),(2024,2027),(2028,2030),(2031,2033)]:
 z=np.array([v for d,v in rows if lo<=d.year<=hi]);print('REG',lo,hi,'n',len(z),'IC %.6f ICIR %.6f'%(z.mean() if len(z) else np.nan,z.mean()/(z.std(ddof=1)+1e-12) if len(z)>1 else np.nan))
q=sig.rank(axis=1,pct=True);print('coverage %.4f turn10 %.4f'%(sig.notna().mean().mean(),np.nanmean((q-q.shift(10)).abs().mean(axis=1))))
