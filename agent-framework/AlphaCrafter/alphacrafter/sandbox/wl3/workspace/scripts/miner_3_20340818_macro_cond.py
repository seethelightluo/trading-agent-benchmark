import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); r=np.log(P).diff(); lp=np.log(P)
# Macro-conditioned downside-adjusted trend: normalise 20d momentum by downside vol;
# in rising-VIX regimes blend with a short (5d) reversal component. All inputs lagged.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(P.index).ffill()
vix_rising=(vix.diff(5)>0).astype(float)
dn=(-r.clip(upper=0)).rolling(30,min_periods=20).std()
trend=(lp-lp.shift(20))/(dn*np.sqrt(20)+1e-8)
rev=-(lp-lp.shift(5))
# cross-sectional rank, conditional regime blend, then lag
f=(trend.rank(axis=1,pct=True)-.5)*(1-vix_rising).values[:,None] + (0.55*(trend.rank(axis=1,pct=True)-.5)+0.45*(rev.rank(axis=1,pct=True)-.5))*vix_rising.values[:,None]
f=f.shift(1)
rows=[]
for h in [1,3,5,10,20]:
 y=lp.shift(-h)-lp
 for dt in f.index:
  a=f.loc[dt]; b=y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,h,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10,20]:
 q=z[z.h==h].ic.dropna(); print('horizon',h,'dates',len(q),'avgN',z[z.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
q=z[z.h==10].set_index('date').ic.dropna()
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turn',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'dates',len(q),'avg instruments',z[z.h==10].n.mean(),'vix rising',vix_rising.mean())
f.to_csv('scripts/miner_3_20340818_macro_cond_signal.csv'); z.to_csv('scripts/miner_3_20340818_macro_cond_ic.csv')
