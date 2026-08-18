import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s,days=6000)
            if x is not None and len(x)>300:return x
        except Exception:pass
raw={s:fetch(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff()
# Five-session reversal, risk-normalized and suppressed in elevated volatility; one-day lag.
rv20=r.rolling(20).std(); vr=rv20/(r.rolling(60).std()+1e-12)
f=(-p.pct_change(5)/(rv20*np.sqrt(5)+1e-12)/(1+vr)).shift(1)
start='2026-07-16'; end='2034-05-24'; rows=[]
for d in f.index:
 y=(p.shift(-60)/p-1).loc[d]; a=pd.concat([f.loc[d],y],axis=1).dropna()
 if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc[start:end]
q.reset_index().to_csv('scripts/miner_2_20340526_short5_vol_reversal_ic.csv',index=False); f.to_csv('scripts/miner_2_20340526_short5_vol_reversal_signal.csv')
print(f'dates={len(q)} avgN={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
for lab,sub in [('early',q.loc[:'2028-12-31']),('mid',q.loc['2029':'2031-12-31']),('recent',q.loc['2032':])]:print(lab,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1))
print('assets',len(raw),'coverage',f.loc[start:end].notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
