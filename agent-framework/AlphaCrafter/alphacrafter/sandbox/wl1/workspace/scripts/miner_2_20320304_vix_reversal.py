import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,days=1800)
 if d is None or len(d)<120:d=get_index_daily_data(s,days=1800)
 if d is not None:rows.append(d[['date','close']].assign(symbol=s))
p=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=np.log(p).diff()
# Macro-conditioned reversal: contrarian 5d shock, scaled by vol, amplified in high-VIX regimes; all lagged.
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']); vc='close' if 'close' in v else v.columns[-1]; vx=v.set_index('date')[vc].reindex(p.index).ffill(); vr=vx/vx.rolling(60).median()-1
f=((-p.pct_change(5)/r.rolling(20).std())*(1+vr.clip(lower=0))).shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; vals=[];ns=[];turn=[];prev=None
 for i in range(len(p)-h):
  z=f.iloc[i];q=y.iloc[i];ok=z.notna()&q.notna()
  if ok.sum()>=8:
   vals.append(z[ok].corr(q[ok]));ns.append(ok.sum());rk=z.rank(pct=True);turn.append(np.nan if prev is None else (rk-prev).abs().mean());prev=rk
 a=np.array([x for x in vals if np.isfinite(x)]);print('horizon',h,'dates',len(a),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,5),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'turn',round(np.nanmean(turn),5))
 if h==10:
  for aa,bb in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-03-04')]:
   q=[]
   for dt in p.loc[aa:bb].index:
    ok=f.loc[dt].notna()&y.loc[dt].notna()
    if ok.sum()>=8:q.append(f.loc[dt][ok].corr(y.loc[dt][ok]))
   q=np.array([x for x in q if np.isfinite(x)]);print('regime',aa[:4],len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20320304_vix_reversal_signal.csv',index=False)
