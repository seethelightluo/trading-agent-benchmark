import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-07-24'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']; px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); v=r.rolling(20,min_periods=15).std()
# medium momentum scaled by own risk, with a continuous market-breadth confirmation
mom=p.pct_change(20).div(v*np.sqrt(20),axis=0)
bread=(r>0).mean(axis=1); confirm=(bread.rolling(20,min_periods=10).mean()-0.5)*2
f=mom.mul((1+0.35*confirm.clip(-1,1)),axis=0)
ics=[]; ns=[]; dates=[]; sig=[]; turns=[]
for i in range(len(p)-10):
 if p.index[i]<p.index[80] or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  ics.append(z);ns.append(ok.sum());dates.append(p.index[i]);sig.append(x)
  if len(sig)>1:
   q=sig[-2]; oo=x.notna()&q.notna(); turns.append((x[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean())
a=np.array(ics);D=pd.DatetimeIndex(dates)
print({'factor':'volnorm_momentum_breadth_20d_10d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(np.array(ns)/15)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turns))})
for n,m in [('180',D>=pd.Timestamp('2030-01-01')),('360',D>=pd.Timestamp('2029-07-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01'))]:
 z=a[m]; print(n,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_1_20300725_volnorm_momentum_breadth_10d_ic.csv',index=False)
pd.DataFrame(sig,index=dates,columns=U).to_csv('scripts/miner_1_20300725_volnorm_momentum_breadth_10d_signal.csv')
