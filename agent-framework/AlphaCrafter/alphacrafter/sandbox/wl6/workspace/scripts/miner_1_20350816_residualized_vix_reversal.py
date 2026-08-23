import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=5000).set_index('date')['close'].astype(float) for s in syms}
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
base=P.pct_change(12).shift(1)/(r.rolling(30).std().shift(1)*np.sqrt(252))
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
q=vix.rolling(252,min_periods=120).rank(pct=True).shift(1)
raw=pd.DataFrame(np.where(q.values[:,None]>=.65,-base.values,base.values),index=P.index,columns=P.columns)
# Residualize cross-sectionally each date against two established reversal directions.
calm=-((P/P.shift(30)-1)/(r.rolling(30).std()+1e-8)+.25*(P/P.rolling(60).max()-1)).shift(1)
bread=(P/P.shift(20)-1)/(r.rolling(30).std().shift(1)+1e-8)
F=pd.DataFrame(index=P.index,columns=P.columns,dtype=float)
for d in P.index:
    y=raw.loc[d]; x=pd.DataFrame({'c':calm.loc[d],'b':bread.loc[d]})
    ok=y.notna()&x.notna().all(axis=1)
    if ok.sum()>=8:
        X=np.column_stack([np.ones(ok.sum()),x.loc[ok].values]); beta=np.linalg.lstsq(X,y[ok].values,rcond=None)[0]
        F.loc[d,ok]=y[ok].values-X@beta
F.to_csv('scripts/miner_1_20350816_residualized_vix_reversal_signal.csv',index_label='date')
for h in [5,10,20,40]:
    fr=P.shift(-h)/P-1; cs=[]; ns=[]; dates=[]
    for d in F.index:
        z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
        if len(z)>=8:
            c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
            if pd.notna(c): cs.append(c);ns.append(len(z));dates.append(d)
    a=pd.Series(cs); print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} start={min(dates).date()} end={max(dates).date()} turnover={F.rank(axis=1,pct=True).diff().abs().mean().mean():.5f}')
