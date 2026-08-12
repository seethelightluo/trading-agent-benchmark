import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(s):
 d=get_stock_daily_data(s, days=4000)
 if d is None or len(d)<150: return None
 d=d.copy(); d['date']=pd.to_datetime(d.date); d=d.set_index('date').sort_index(); return d.close.astype(float)
px={s:load(s) for s in U}; px={s:x for s,x in px.items() if x is not None}
wide=pd.DataFrame(px).sort_index(); r=wide.pct_change()
# beta-neutral residual medium trend, risk adjusted by downside volatility
mkt=wide['SPX'].pct_change()
rows=[]
for s in px:
 if s=='SPX': beta=pd.Series(1.,index=wide.index)
 else: beta=r[s].rolling(60,min_periods=40).cov(mkt)/mkt.rolling(60,min_periods=40).var()
 resid=r[s]-beta*mkt
 trend=resid.rolling(20,min_periods=15).sum()
 down=r[s].where(r[s]<0,0).rolling(40,min_periods=25).std()
 sig=trend/(down*np.sqrt(252)+0.02)
 rows.append(sig.rename(s))
f=pd.concat(rows,axis=1).shift(1)
for h in [1,5,10,20]:
 fr=wide.pct_change(h).shift(-h)
 ics=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1])); ns.append(len(z))
 a=np.array(ics); print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),6),'hit',round(np.mean(a>0),4))
# turnover and coverage
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean(); cov=f.notna().sum(axis=1).mean()/len(U)
print('turnover',round(turn,6),'coverage',round(cov,6),'rows',len(f),'assets',len(U),'last',f.index.max().date())
out=Path('scripts/miner_1_20300404_residual_beta_downside_trend_signal.csv'); out.parent.mkdir(exist_ok=True)
f.to_csv(out); print('artifact',out)
# recent 2029 onward 1d
fr=wide.pct_change().shift(-1); aa=[]
for dt in f.loc['2029-01-01':].index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1]))
a=np.array(aa); print('recent dates',len(a),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),6))
