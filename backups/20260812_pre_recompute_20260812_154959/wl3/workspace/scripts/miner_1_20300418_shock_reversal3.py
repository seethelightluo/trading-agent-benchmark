import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<150:d=get_index_daily_data(s,3000)
 if d is not None and len(d):P[s]=d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index();lr=np.log(px).diff();down=lr.where(lr<0,0).rolling(30,min_periods=15).std()*np.sqrt(30)
# nonlinear shock reversal: reward larger recent losses, normalized by downside risk; lagged
sig=(-(px.pct_change(3))*px.pct_change(3).abs()/down).replace([np.inf,-np.inf],np.nan).shift(1)
def calc(h):
 f=px.shift(-h).div(px)-1;R=[]
 for t in sig.index:
  z=pd.concat([sig.loc[t],f.loc[t]],axis=1).dropna()
  if len(z)>=8:R.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 return pd.Series(R)
R=calc(1);print('dates',len(px),'instruments',len(P),'obs',len(R),'avg_n',round(sig.notna().sum(axis=1).mean(),2),'coverage',round(sig.notna().sum(axis=1).mean()/len(U),4))
for lab,q in [('full',R),('2020-22',R.iloc[:700]),('2023-25',R.iloc[700:1400]),('2026-27',R.iloc[1400:2100]),('2028+',R.iloc[2100:]),('recent250',R.tail(250))]:print(lab,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [3,5,10]:q=calc(h);print('decay',h,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'n',len(q))
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4));out='scripts/miner_1_20300418_shock_reversal3_signal.csv';sig.to_csv(out,index_label='date');print('artifact',out)
