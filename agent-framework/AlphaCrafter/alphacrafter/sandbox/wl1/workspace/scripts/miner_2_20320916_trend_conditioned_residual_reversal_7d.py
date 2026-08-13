import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=3000) for s in U}
p=pd.concat({s:d.set_index('date').close for s,d in D.items() if d is not None},axis=1).sort_index().ffill(); r=p.pct_change();
# vectorized rolling dispersion and per-date cross-sectional residual signal
asset_vol=r.rolling(20).std(); disp=r.rolling(20).std().mean(axis=1); med=disp.rolling(120).median()
ret7=p.pct_change(7); resid=ret7.sub(ret7.mean(axis=1),axis=0); f=-resid/asset_vol
active=((p.pct_change(20).mean(axis=1)<0)|(disp>med)); f=f.mul(np.where(active,1,.25),axis=0)
fr=p.pct_change(10).shift(-10); rows=[]; sig=[]
for dt in p.index:
 z=pd.concat([f.loc[dt].rename('f'),fr.loc[dt].rename('r')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.f.corr(z.r),len(z),bool(active.loc[dt]))); sig.append(f.loc[dt].rename(dt))
o=pd.DataFrame(rows,columns=['date','ic','n','active']).set_index('date'); print('dates',len(o),'avg_n',o.n.mean(),'active_frac',o.active.mean(),'coverage',f.notna().stack().mean()); print('IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(ddof=1),'hit',(o.ic>0).mean())
for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-09-15')]:
 q=o.loc[a:b]; print(a[:4]+'-'+b[:4],len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean())
ss=pd.DataFrame(sig); print('rank_turnover',ss.rank(pct=True).diff().abs().mean(axis=1).mean()); ss.to_csv('scripts/miner_2_20320916_trend_conditioned_residual_reversal_7d_signal.csv')
