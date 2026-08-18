import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<120:d=get_index_daily_data(s,5000)
 if d is not None and len(d)>120:px[s]=d.set_index(pd.to_datetime(d.date)).close
P=pd.DataFrame(px).sort_index().ffill(); R=P.pct_change()
# Long-horizon momentum normalized by recent risk, lagged one bar.
F=(P.pct_change(60)/(R.rolling(40).std()*np.sqrt(40)+1e-8)).shift(1)
FR=P.shift(-10)/P-1; out=[]
for dt in F.index:
 x,y=F.loc[dt],FR.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:out.append((dt,x[ok].corr(y[ok]),ok.sum()))
q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); print('universe',len(U),'usable',len(px),'dates',len(q),'avg_n',q.n.mean());print('IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
for a,b in [('2026-07-16','2029-12-31'),('2030-01-01','2033-12-31'),('2029-01-01','2033-12-31'),('2033-01-01','2034-02-03')]:
 z=q.loc[a:b].ic;print(a,b,'n',len(z),'IC %.8f ICIR %.8f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('coverage',F.notna().sum().sum()/(F.shape[0]*F.shape[1]),'turnover',F.rank(pct=True).diff().abs().mean(axis=1).mean());q.to_csv('scripts/miner_1_20340203_risk_momentum_60d_ic.csv');F.to_csv('scripts/miner_1_20340203_risk_momentum_60d_signal.csv')
