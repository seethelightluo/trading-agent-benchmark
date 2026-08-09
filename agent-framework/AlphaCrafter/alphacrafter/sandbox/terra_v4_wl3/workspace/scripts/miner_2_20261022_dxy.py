import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-10-22')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in assets}
# DXY-neutral momentum: 20-session asset return residualized against contemporaneous DXY return,
# estimated only with completed observations and a 60-session rolling beta.
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().close
dxy=dxy[dxy.index<=end]
prices=pd.concat({a:D[a].close[D[a].index<=end] for a in assets},axis=1).sort_index()
r=prices.pct_change(); zd=dxy.pct_change().reindex(r.index)
fac=pd.DataFrame(index=prices.index,columns=assets,dtype=float)
for a in assets:
    x=r[a]; beta=x.rolling(60,min_periods=40).cov(zd)/zd.rolling(60,min_periods=40).var().replace(0,np.nan)
    resid=x-beta*zd
    fac[a]=resid.rolling(20,min_periods=15).sum()
for h in [1,5,10]:
    fwd=prices.pct_change(h).shift(-h); ics=[]; dates=[]; ns=[]
    for dt in fac.index:
        x=fac.loc[dt].dropna(); y=fwd.loc[dt].reindex(x.index).dropna(); x=x.reindex(y.index)
        if len(x)>=8 and x.nunique()>1 and y.nunique()>1:
            q=spearmanr(x,y).statistic
            if np.isfinite(q): ics.append(q);dates.append(dt);ns.append(len(x))
    s=pd.Series(ics,index=dates); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std(),5),'hit',round((s>0).mean(),4))
    if h==1:
      for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
       q=s[(s.index.year>=lo)&(s.index.year<=hi)];print('REG',lo,hi,len(q),round(q.mean(),5),round(q.mean()/q.std(),5))
print('coverage',round(fac.notna().sum(axis=1).mean()/15,4),'turnover',round(fac.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20261022_dxy_signal.csv',index=False)
