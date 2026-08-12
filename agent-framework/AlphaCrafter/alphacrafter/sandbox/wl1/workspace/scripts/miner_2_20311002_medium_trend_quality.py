import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# medium-term trend quality: 60d momentum, penalize downside risk, gated by broad market breadth

def fetch(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s,days=5000)
            if x is not None and len(x): return x[['date','close']].copy()
        except Exception: pass
    return None
from alphacrafter.sim.utils import get_stock_daily_data
px={s:fetch(s) for s in U}
px={s:x for s,x in px.items() if x is not None}
close=pd.concat([d.rename(columns={'close':s}).set_index('date') for s,d in px.items()],axis=1).sort_index().ffill()
ret=np.log(close).diff()
# use only information through t, and score is lagged one session
mom= np.log(close/close.shift(60))
print('nan close',close.isna().mean().to_dict(),'momvalid',mom.notna().sum().sum(),'retvalid',ret.notna().sum().sum())
down=ret.where(ret<0,0.0).pow(2).rolling(60,min_periods=20).mean().pow(0.5)
quality=mom/(down+1e-8)
breadth=(mom>0).mean(axis=1)
# smooth breadth gate, avoids macro external series and is interpretable
f=quality.mul(0.70+0.60*breadth.clip(0,1),axis=0)
f=f.shift(1)
print('fvalid',f.notna().sum().sum(), 'quality',quality.notna().sum().sum(), 'breadth',breadth.notna().sum())
rows=[]
for h in [1,5,10,20]:
  fr=close.pct_change(h).shift(-h)
  ics=[]
  for dt in f.index:
    x=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
      ics.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'))
  a=np.array(ics); mean=float(np.nanmean(a)); sd=float(np.nanstd(a,ddof=1));
  print('H',h,'dates',len(a),'meanIC',round(mean,6),'ICIR',round(mean/(sd+1e-12)*np.sqrt(252),6),'hit',round(float((a>0).mean()),4))
# turnover based rank changes
r=f.rank(axis=1,pct=True); turnover=float((r.diff().abs().mean(axis=1).dropna()).mean())
print('available',list(px), 'close_shape',close.shape)
print('rows',len(close),'dates',len(f.dropna(how='all')),'avg_assets',float(f.notna().sum(axis=1).mean()),'coverage',float(f.notna().mean().mean()),'turnover',turnover)
# regime 20d
fr=close.pct_change(20).shift(-20)
for label,a,b in [('2020-22','2020','2022'),('2023-25','2023','2025'),('2026-28','2026','2028'),('2029-30','2029','2030'),('2031','2031','2031')]:
 vals=[]
 for dt in f.loc[a:b].index:
  x=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1: vals.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'))
 a=np.array(vals)
 print('REG',label,len(a),round(float(np.nanmean(a)),6),round(float(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12)*np.sqrt(252),6)) if len(a)>1 else None)
# artifact for deterministic audit
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20311002_medium_trend_quality_signal.csv',index=False)
