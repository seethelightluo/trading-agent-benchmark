import numpy as np,pandas as pd,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close for s in U},axis=1,sort=True)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].rename('DXY')
P=P.join(dxy,how='left').loc[:'2026-10-21']; r=np.log(P).diff(); x=r['DXY']; var=x.rolling(60,min_periods=40).var()
f=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U:
    q=r[s]; beta=q.rolling(60,min_periods=40).cov(x)/var
    # 20-session momentum after removing exposure to contemporaneous DXY move
    sig=q.rolling(20,min_periods=15).sum()-beta*x.rolling(20,min_periods=15).sum()
    f.loc[sig.index,s]=sig
f.to_csv('scripts/miner_1_20261022_dxy_residual_momentum_signal.csv')
for h in [1,5,10]:
    fw=np.log(P[U]).shift(-h)-np.log(P[U]); z=[];ns=[];ds=[]
    for dt in f.index:
        a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
        if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1:
            z.append(spearmanr(a.f,a.r).statistic);ns.append(len(a));ds.append(dt)
    s=pd.Series(z,index=pd.DatetimeIndex(ds)); print(f'h={h} dates={len(s)} avgN={np.mean(ns):.2f} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1):.6f} hit={(s>0).mean():.4f}')
    if h==1:
        print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
        for label,mask in [('2020-22',s.index<='2022-12-31'),('2023-24',(s.index>='2023-01-01')&(s.index<='2024-12-31')),('2025-26',s.index>='2025-01-01')]:
            q=s[mask];print(label,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
print('period',P.index.min(),P.index.max(),'assets',len(U),'signal_artifact scripts/miner_1_20261022_dxy_residual_momentum_signal.csv')
