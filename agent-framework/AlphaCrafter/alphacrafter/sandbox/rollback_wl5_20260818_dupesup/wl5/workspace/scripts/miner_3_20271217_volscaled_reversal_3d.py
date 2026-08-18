import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
C=pd.DataFrame({s:D[s]['close'] for s in U})
cut=pd.Timestamp('2027-12-17')
ret3=C.pct_change(3); vol20=C.pct_change().rolling(20).std()
raw=-(ret3.sub(ret3.median(axis=1),axis=0))/vol20
for h in [1,3,5,10]:
    vals=[]; dates=[]; counts=[]
    Y=C.shift(-h)/C-1
    for d in raw.index:
        if d>cut: continue
        g=pd.DataFrame({'f':raw.loc[d],'y':Y.loc[d]}).replace([np.inf,-np.inf],np.nan).dropna()
        if d>=pd.Timestamp('2020-01-01') and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
            vals.append(spearmanr(g.f,g.y).statistic); dates.append(d); counts.append(len(g))
    z=np.asarray(vals); recent=z[np.asarray(dates)>=pd.Timestamp('2026-07-16')]
    print('horizon',h,'dates',len(z),'avg_n',round(np.mean(counts),2),'coverage',round(np.mean(counts)/15,4),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round(np.mean(z>0),4),'recent_dates',len(recent),'recent_IC',round(recent.mean(),6) if len(recent) else None,'recent_ICIR',round(recent.mean()/recent.std(ddof=1),6) if len(recent)>1 else None)
    if h==10:
        pd.DataFrame({'date':dates,'ic':vals}).to_csv('scripts/miner_3_20271217_volscaled_reversal_3d_ic.csv',index=False)
