import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
u=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in u:
    d=None
    try: d=get_index_daily_data(s, days=4200)
    except Exception: pass
    if d is None:
      try: d=get_stock_daily_data(s,days=4200)
      except Exception: pass
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); prices[s]=x.set_index('date').close
p=pd.DataFrame(prices).sort_index(); r=p.pct_change()
mom=p.pct_change(20); vol=r.rolling(30,min_periods=20).std(); down=r.where(r<0,0).rolling(30,min_periods=20).std()
sig=(mom/(vol+1e-12))*(vol/(down+vol+1e-12))
disp=r.rolling(20,min_periods=15).std().median(axis=1)
sig=sig.div((1+disp).replace(0,np.nan),axis=0).shift(1)
for h in [1,5,10,20]:
    fwd=p.shift(-h).div(p)-1; vals=[]; nins=[]
    for dt in sig.index:
        z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            c=z.iloc[:,0].corr(z.iloc[:,1])
            if np.isfinite(c): vals.append(c); nins.append(len(z))
    a=np.asarray(vals); ic=np.mean(a); icir=ic/np.std(a,ddof=1)
    print('H',h,'IC %.6f dailyICIR %.6f hit %.3f dates %d avgN %.2f'%(ic,icir,np.mean(a>0),len(a),np.mean(nins)))
    if h==20:
      for n in [252,756,1260]:
        q=a[-n:] if len(a)>n else a; print(' recent',n,'IC %.6f ICIR %.6f'%(np.mean(q),np.mean(q)/np.std(q,ddof=1)))
print('assets',len(p.columns),list(p.columns),'rows',len(p),'range',p.index.min(),p.index.max())
sig.to_csv('scripts/miner_2_20350709_risk_adjusted_momentum_signal.csv',index_label='date')
