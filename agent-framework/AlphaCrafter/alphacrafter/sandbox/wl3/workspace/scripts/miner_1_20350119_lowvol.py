import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
syms=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 if d is not None and len(d):px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Lagged low-volatility effect, with downside risk penalty: lower downside deviation ranks higher.
down=R.clip(upper=0).rolling(30).std(); total=R.rolling(30).std()
f=(-(0.7*total+0.3*down)).shift(1)
ics=[]; nv=[]; tv=[]
for i in range(31,len(P)-10):
 n=f.columns[f.iloc[i].notna()&P.iloc[i].notna()&P.iloc[i+10].notna()]
 if len(n)<8:continue
 a=f.iloc[i][n]; y=P.iloc[i+10][n]/P.iloc[i][n]-1
 ics.append((P.index[i],a.corr(y,method='spearman')));nv.append(len(n))
 if i>31:tv.append(np.mean(abs(a.rank(pct=True)-f.iloc[i-1][n].rank(pct=True))))
s=pd.Series(dict(ics)).dropna()
for l,z in [('all',s),('recent120',s.tail(120)),('recent252',s.tail(252)),('recent504',s.tail(504))]:print(l,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('instruments',len(syms),'avg_valid',round(np.mean(nv),3),'coverage',round(np.mean(nv)/len(syms),4),'turnover',round(np.mean(tv),4),'period',P.index[0],P.index[-1])
for j,z in enumerate(np.array_split(s,4),1):print('quartile',j,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
pd.DataFrame({'date':s.index,'factor_ic':s.values}).to_csv('scripts/miner_1_20350119_lowvol_signal.csv',index=False)
