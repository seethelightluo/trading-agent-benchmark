import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    try: d=get_stock_daily_data(s, days=6000)
    except Exception:
        try: d=get_index_daily_data(s, days=6000)
        except Exception: d=None
    if d is not None and len(d)>150:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); frames[s]=d.sort_values('date').set_index('date')
px=pd.DataFrame({s:d['close'] for s,d in frames.items()}).sort_index(); ret=np.log(px).diff()
mom=np.log(px/px.shift(60)); down=ret.where(ret<0,0).rolling(60,min_periods=30).std()*np.sqrt(252)
f=(mom/down.replace(0,np.nan)).shift(1).replace([np.inf,-np.inf],np.nan); fwd=np.log(px.shift(-10)/px)
ics=[]; dates=[]; nms=[]
for dt in f.index:
 a=f.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: ics.append(a[ok].corr(b[ok],method='spearman')); dates.append(dt); nms.append(ok.sum())
ics=pd.Series(ics,index=pd.to_datetime(dates)).dropna(); r=f.rank(axis=1,pct=True); to=r.diff().abs().mean(axis=1).dropna()
print('candidate tail_risk_adjusted_momentum_60d'); print('data_dates',px.index.min(),px.index.max(),'assets',len(frames)); print('valid_dates',len(ics),'avg_names',np.mean(nms),'coverage',np.mean(nms)/15)
print('IC %.9f ICIR %.6f hit %.4f turnover %.6f'%(ics.mean(),ics.mean()/ics.std(ddof=1)*np.sqrt(len(ics)),(ics>0).mean(),to.mean()))
for h in [5,10,20,40]:
 ff=np.log(px.shift(-h)/px); z=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&ff.loc[dt].notna()
  if ok.sum()>=8:z.append(f.loc[dt][ok].corr(ff.loc[dt][ok],method='spearman'))
 z=pd.Series(z).dropna(); print('decay',h,round(z.mean(),9),len(z))
for a,b in [('2020-01-01','2027-12-31'),('2028-01-01','2031-12-31'),('2032-01-01','2035-03-28')]:
 z=ics[(ics.index>=a)&(ics.index<=b)]; print('regime',a,b,round(z.mean(),9),len(z))
f.to_csv('scripts/miner_2_20350329_tail_risk_adjusted_momentum_signal.csv')
