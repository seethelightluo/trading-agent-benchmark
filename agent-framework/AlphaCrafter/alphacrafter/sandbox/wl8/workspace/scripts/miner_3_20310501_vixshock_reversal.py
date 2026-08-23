import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    d=get_stock_daily_data(s,2500)
    if d is None or len(d)<300: d=get_index_daily_data(s,2500)
    return d
px={s:fetch(s) for s in U}
px={s:d.set_index(pd.to_datetime(d.date)).close.astype(float) for s,d in px.items() if d is not None}
# local observation-only VIX, align by date; no contemporaneous use (shift one)
v=pd.read_csv('../persistent/index_data/VIX.csv')
v['date']=pd.to_datetime(v['date']); v=v.set_index('date')['close'].astype(float)
dates=sorted(set.intersection(*[set(x.index) for x in px.values()]) & set(v.index))
D=pd.DataFrame({s:px[s].reindex(dates) for s in U}); vv=v.reindex(dates)
ret=D.pct_change(); vshock=vv.pct_change(5).shift(1); vz=(vshock-vshock.rolling(120).mean())/vshock.rolling(120).std()
# candidate: reversal amplified only when lagged VIX shock positive, smoothly bounded
sig=-(ret.rolling(5).sum()).shift(1) * (1+0.75*np.tanh(vz.fillna(0)))[:,None] if False else None
# correction broadcast
mult=(1+0.75*np.tanh(vz.fillna(0))).values[:,None]
sig=-(ret.rolling(5).sum().shift(1).values)*mult
S=pd.DataFrame(sig,index=dates,columns=U)
for h in [1,5,10]:
    fwd=D.pct_change(h).shift(-h)
    ics=[]
    for dt in dates:
        z=pd.concat([S.loc[dt],fwd.loc[dt]],axis=1).dropna()
        if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    a=np.array(ics); print('horizon',h,'dates',len(a),'avg_n',15,'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1)*np.sqrt(1) if np.nanstd(a,ddof=1)>0 else 0,(a>0).mean()))
# turnover based rank signal
r=S.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean().mean(),'coverage',S.notna().mean().mean(),'range',dates[0],dates[-1])
# artifact for 10d
out=S.copy(); out.index.name='date'; out.to_csv('scripts/miner_3_20310501_vixshock_reversal_signal.csv')
