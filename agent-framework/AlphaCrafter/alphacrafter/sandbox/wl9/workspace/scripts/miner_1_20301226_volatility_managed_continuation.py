import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-12-26')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]
r=p.pct_change()
# Volatility-managed continuation: recent 10d directional return scaled by trailing
# 30d realized risk; lagged one day to avoid look-ahead.
f=(r.rolling(10).sum()/r.rolling(30).std()).shift(1)
# forward returns over several horizons, and daily IC observations
outs={}
for h in [1,5,10,20,40,60]:
    fr=p.shift(-h)/p-1
    vals=[]
    for dt in f.index:
        x=f.loc[dt]; y=fr.loc[dt]
        z=pd.concat([x,y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    a=np.array(vals); outs[h]=(len(a),np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)/np.sqrt(len(a))),np.mean(a>0))
print('end',end.date(),'dates',len(p),'assets',len(U),'avg valid',f.notna().sum(axis=1).mean())
for h,v in outs.items(): print('H',h,'n',v[0],'IC %.6f ICIR %.6f hit %.4f'%v[1:])
# regime windows
for label,a,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-26')]:
    fr=p.shift(-10)/p-1; vals=[]
    for dt in f.loc[a:b].index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    q=np.array(vals); print('REG',label,len(q),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(q),np.mean(q)/(np.std(q,ddof=1)/np.sqrt(len(q))),np.mean(q>0)))
# turnover proxy using cross-sectional ranks
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean()
print('coverage %.6f turnover_proxy %.6f'%(f.notna().mean().mean(),turn))
# signal artifact for provenance
out=f.reset_index().rename(columns={'index':'date'}); out.to_csv('scripts/miner_1_20301226_volatility_managed_continuation_signal.csv',index=False)
