import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

acct=get_account_dict(); syms=acct.get('watch_list',[])
if not syms: syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
    d=get_stock_daily_data(s, days=6000)
    if d is not None and len(d)>100:
        x=d.copy(); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').set_index('date')
        px[s]=x['close'].astype(float)
p=pd.DataFrame(px).sort_index(); rets=p.pct_change()
# persistence: signed 20d momentum multiplied by fraction of up days, volatility scaled
mom=p/p.shift(20)-1
up=(rets>0).rolling(20).mean()
vol=rets.rolling(20).std()
f=(mom*(2*up-1)/(vol+1e-8)).shift(1)
ics=[]; ns=[]; turnovers=[]; prev=None
for i in range(1,len(p)-10):
    sig=f.iloc[i]; fw=p.iloc[i+10]/p.iloc[i]-1
    z=pd.concat([sig.rename('s'),fw.rename('r')],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.s,z.r).statistic
        if np.isfinite(ic):
            ics.append(ic); ns.append(len(z))
            ranks=z.s.rank()/len(z)
            if prev is not None: turnovers.append(np.mean(abs(ranks-prev)))
            prev=ranks
print('dates',len(ics),'avgN',np.mean(ns),'coverage',len(ics)/max(1,len(p)-11))
print('IC %.8f ICIR %.8f hit %.4f turnover %.8f'%(np.mean(ics),np.mean(ics)/(np.std(ics,ddof=1)+1e-12),np.mean(np.array(ics)>0),np.mean(turnovers)))
for h in [1,5,10,20]:
  a=[]
  for i in range(1,len(p)-h):
    z=pd.concat([f.iloc[i].rename('s'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).dropna()
    if len(z)>=8: a.append(spearmanr(z.s,z.r).statistic)
  print('decay',h, np.mean(a),len(a))
for cut in [365,750,1260]:
 print('recent',cut,np.mean(ics[-cut:]),np.mean(ics[-cut:])/(np.std(ics[-cut:],ddof=1)+1e-12))
