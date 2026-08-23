import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# A compact, interpretable factor: 20-session return, penalized by recent volatility and reversals.
# lag one completed session; forward returns begin after signal date.
xs={}
for s in U:
    d=get_stock_daily_data(s, days=6000)
    if d is None or len(d)<100: continue
    x=d.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index()
    p=pd.to_numeric(x['close'],errors='coerce'); r=np.log(p).diff()
    mom=np.log(p/p.shift(20))
    vol=r.rolling(20).std()*np.sqrt(252)
    # consistency rewards positive return days, but only over the recent short window
    cons=r.rolling(20).mean()/r.rolling(20).std()
    # Penalize sharp one-day reversal risk via downside deviation
    down=np.sqrt((r.clip(upper=0)**2).rolling(20).mean())*np.sqrt(252)
    f=(mom/vol)*(1+0.35*cons)/(1+down)
    xs[s]=pd.DataFrame({'f':f.shift(1),'p':p})

all_dates=sorted(set().union(*[set(x.index) for x in xs.values()]))
ics=[]; rows=[]; horizons=[5,10,20,40]
for dt in all_dates:
    vals=[]; fs=[]
    for s,x in xs.items():
        if dt not in x.index: continue
        i=x.index.get_loc(dt); f=x.iloc[i]['f']
        if i+10>=len(x) or not np.isfinite(f): continue
        vals.append(s); fs.append(f)
    # use each asset's next h observations, requiring exact dates in its own series
    for h in horizons:
        a=[]; b=[]
        for s in vals:
            x=xs[s]; i=x.index.get_loc(dt)
            if i+h>=len(x): continue
            fr=np.log(x.iloc[i+h]['p']/x.iloc[i]['p'])
            if np.isfinite(fr) and np.isfinite(x.iloc[i]['f']): a.append(x.iloc[i]['f']); b.append(fr)
        if len(a)>=8: rows.append((dt,h,len(a),np.corrcoef(a,b)[0,1]))
R=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('assets',len(xs),'dates',R.date.nunique(),'range',R.date.min(),R.date.max())
for h in horizons:
 q=R[R.h==h].dropna(); ic=q.ic
 print('H',h,'obs',len(q),'avgN',q.n.mean(),'IC %.8f'%ic.mean(),'ICIR %.6f'%(ic.mean()/ic.std()),'hit %.4f'%((ic>0).mean()))
# rolling recent and broad regimes
for label,lo,hi in [('early','2020-01-01','2027-12-31'),('mid','2028-01-01','2031-12-31'),('recent','2032-01-01','2035-04-11')]:
 q=R[(R.h==10)&(R.date>=lo)&(R.date<=hi)].dropna(); print(label,len(q),'IC %.8f'%q.ic.mean(),'ICIR %.5f'%(q.ic.mean()/q.ic.std()) if len(q)>1 else np.nan)
# coverage and factor turnover (rank changes among common assets)
print('coverage',sum(np.isfinite(x.f).sum() for x in xs.values())/sum(len(x) for x in xs.values()))
print('turnover proxy',np.mean([np.mean(np.abs(np.sign(x.f.diff()).dropna())) for x in xs.values()]))
# save signal artifact for chosen horizon
out=[]
for s,x in xs.items():
 for dt,row in x.iterrows():
  if np.isfinite(row.f): out.append({'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':float(row.f)})
pd.DataFrame(out).to_csv('scripts/miner_1_20350412_short_horizon_trend_quality_signal.csv',index=False)
