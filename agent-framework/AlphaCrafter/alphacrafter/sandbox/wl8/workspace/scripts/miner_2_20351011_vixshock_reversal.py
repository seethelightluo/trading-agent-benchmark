import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# retrieve broad history available as of runtime
px={s:(get_stock_daily_data(s,5000) if __import__('os').path.exists('../persistent/stock_data/'+s+'.csv') else get_index_daily_data(s,5000)) for s in U}
vix=get_index_daily_data('VIX',5000)
frames=[]
for s,d in px.items():
 if d is None or len(d)<100: continue
 x=d[['date','close']].copy(); x['asset']=s; x=x.set_index('date')['close'].rename(s); frames.append(x)
P=pd.concat(frames,axis=1).sort_index().ffill()
if vix is None: raise RuntimeError('no VIX')
V=vix.set_index('date')['close'].reindex(P.index).ffill()
r=np.log(P).diff(); rv=r.rolling(20).std(); ret5=np.log(P).diff(5)
# completed-day signal: VIX shock measured at t, used after shift; contrarian 5d return normalized
shock=(V.pct_change(5)>0.08) & (V>V.rolling(120).median())
f=(-ret5/rv).where(shock,0).shift(1)
fwd=np.log(P).shift(-10)-np.log(P)
ics=[]; turnover=[]; ns=[]
for dt in f.index:
 a=f.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  ics.append(a[ok].corr(b[ok],method='spearman')); ns.append(ok.sum())
  # rank turnover against previous date, only among valid
  if len(turnover)==0: pass
  else:
   prev=f.shift(1).loc[dt]; oo=ok&prev.notna()
   if oo.sum()>=8: turnover.append((a[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean())
ics=pd.Series(ics).dropna(); icir=ics.mean()/ics.std(ddof=1)*np.sqrt(252) if len(ics)>1 else np.nan
print({'dates':len(ics),'avgN':float(np.mean(ns)),'coverage':float(f.notna().sum().sum()/f.size),'mean_daily_IC':float(ics.mean()),'daily_paper_ICIR':float(icir),'hit':float((ics>0).mean()),'turnover':float(np.mean(turnover)),'shock_days':int(shock.sum())})
for n in [365,750,1260]: print('recent',n,float(ics.tail(n).mean()) if len(ics)>=n else None)
for h in [1,5,10,20]:
 ff=np.log(P).shift(-h)-np.log(P); z=[]
 for dt in f.index:
  a=f.loc[dt]; b=ff.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: z.append(a[ok].corr(b[ok],method='spearman'))
 print('decay',h,float(pd.Series(z).dropna().mean()),len(z))
# signal artifact for provenance
out=f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20351011_vixshock_reversal_signal.csv',index=False)
