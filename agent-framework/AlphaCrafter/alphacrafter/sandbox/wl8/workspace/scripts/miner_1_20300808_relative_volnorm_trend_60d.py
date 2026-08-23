import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-08-08'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']; px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); v=r.rolling(30,min_periods=20).std()
# Relative medium-horizon trend: asset 60d return minus contemporaneous cross-sectional median, risk scaled.
r60=p.pct_change(60); rel=r60.sub(r60.median(axis=1),axis=0); f=rel.div(v,axis=0)
ics=[]; ns=[]; dates=[]; sig=[]; turns=[]
for i in range(len(p)-10):
 if p.index[i] < pd.Timestamp('2026-07-16') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  ics.append(z); ns.append(ok.sum()); dates.append(p.index[i]); sig.append(x)
  if len(sig)>1:
   q=sig[-2]; oo=x.notna()&q.notna(); turns.append((x[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean())
a=np.array(ics); D=pd.DatetimeIndex(dates)
print({'factor':'relative_volnorm_trend_60d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(np.array(ns)/15)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turns))})
for name,m in [('180',D>=pd.Timestamp('2030-01-01')),('360',D>=pd.Timestamp('2029-08-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01'))]:
 z=a[m]; print(name,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
out=pd.DataFrame(sig,index=dates,columns=U); out.index.name='date'; out.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_1_20300808_relative_volnorm_trend_60d_signal.csv',index=False)
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_1_20300808_relative_volnorm_trend_60d_ic.csv',index=False)
