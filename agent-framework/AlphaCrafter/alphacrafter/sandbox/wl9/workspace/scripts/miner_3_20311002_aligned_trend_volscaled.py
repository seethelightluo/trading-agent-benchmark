import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def fetch(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(symbol=s, days=5000)
            if x is not None and len(x)>300: return x[['date','close']].copy()
        except Exception: pass
    return None

def main():
    px={s:fetch(s) for s in UNIV}; px={s:x for s,x in px.items() if x is not None}
    close=pd.concat([x.set_index('date')['close'].rename(s) for s,x in px.items()],axis=1).sort_index().ffill()
    lr=np.log(close).diff()
    # Aligned trend: medium-term return, confirmed by long-term direction,
    # normalized by recent volatility; lagged naturally by using t signal for t+forward.
    r20=close/close.shift(20)-1
    r60=close/close.shift(60)-1
    vol20=lr.rolling(20).std()*np.sqrt(252)
    sig=(r20*np.sign(r60)/vol20).replace([np.inf,-np.inf],np.nan)
    # require alignment, otherwise neutral rather than forcing a direction
    sig=sig.where(np.sign(r20)==np.sign(r60),0.0)
    rows=[]
    for h in [5,10,20,40,60]:
        fwd=close.shift(-h)/close-1
        vals=[]
        for d in sig.index:
            z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
            if len(z)>=8: vals.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
        q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
        ic=q.ic
        rows.append((h,len(q),q.n.mean(),ic.mean(),ic.std(ddof=1),ic.mean()/ic.std(ddof=1)*np.sqrt(252) if ic.std(ddof=1)>0 else np.nan,(ic>0).mean()))
    print('assets',len(px),'dates',len(close),'range',close.index.min(),close.index.max())
    print('horizon dates avg_n IC ICIR_daily_annualized hit')
    for x in rows: print('%d %d %.2f %+.6f %+.6f %+.6f %.4f'%x)
    # explicit daily paper ICIR convention annualized sqrt(252), and regime blocks at selected 20d horizon
    h=20; fwd=close.shift(-h)/close-1; vals=[]
    for d in sig.index:
        z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
        if len(z)>=8: vals.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
    q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
    for name,a,b in [('2024-26','2024-01-01','2026-12-31'),('2027-29','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31'),('2031YTD','2031-01-01','2031-10-02')]:
        u=q.loc[a:b,'ic']; print('regime',name,len(u),u.mean(),u.mean()/u.std(ddof=1)*np.sqrt(252) if len(u)>2 and u.std(ddof=1)>0 else np.nan)
    # turnover of cross-sectional rank weights, and coverage
    ranks=sig.rank(axis=1,pct=True); ch=ranks.diff().abs().mean(axis=1)
    sig.to_csv('scripts/miner_3_20311002_aligned_trend_volscaled_signal.csv',index_label='date'); print('coverage',sig.notna().mean().mean(),'turnover_proxy',ch.dropna().mean()); print('signal_artifact','scripts/miner_3_20311002_aligned_trend_volscaled_signal.csv')

if __name__=='__main__': main()
