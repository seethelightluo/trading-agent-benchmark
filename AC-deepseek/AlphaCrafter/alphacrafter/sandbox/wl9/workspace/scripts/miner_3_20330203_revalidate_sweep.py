"""miner_3 cycle 2033-02-03: re-validate effective factors + sweep new candidates.
Visible through 2033-02-02. No lookahead. Admission gates: abs daily paper IC
>=0.0070, abs ICIR>=0.084 (10d horizon). Warm-up only.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import json

VISIBLE_END = '2033-02-02'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets, end):
    closes, highs, lows, vols = {}, {}, {}, {}
    for a in assets:
        f = STOCK_DIR / f'{a}.csv'
        if not f.exists(): f = INDEX_DIR / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date']<=end].sort_values('date').set_index('date')
        closes[a]=df['close'].astype(float); highs[a]=df['high'].astype(float)
        lows[a]=df['low'].astype(float)
        vols[a] = df['volume'].astype(float) if 'volume' in df else pd.Series(np.nan, index=df.index)
    return closes, highs, lows, vols

closes,highs,lows,vols = load(ASSETS, VISIBLE_END)
close=pd.DataFrame(closes).dropna()
high=pd.DataFrame(highs).reindex(close.index); low=pd.DataFrame(lows).reindex(close.index)
vol=pd.DataFrame(vols).reindex(close.index)
rets=close.pct_change().dropna()
fwd5 =rets.shift(-5).rolling(5).mean(); fwd10=rets.shift(-10).rolling(10).mean(); fwd20=rets.shift(-20).rolling(20).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}", flush=True)

def mac(c):
    df=pd.read_csv(INDEX_DIR/f'{c}.csv',parse_dates=['date']); df=df[df['date']<=VISIBLE_END].set_index('date')['close'].astype(float); return df
vix=mac('VIX'); dxy=mac('DXY'); usdcny=mac('USDCNY'); usdjpy=mac('USDJPY'); eurusd=mac('EURUSD')
dVIX=vix.pct_change(); dCNY=usdcny.pct_change(); dJPY=usdjpy.pct_change(); dDXY=dxy.pct_change()
dUSDNY = dCNY  # usdcny change for cny_beta

def compute_ic(fv,fwd,min_dates=30,start=None):
    f=fv.reindex(fwd.index); idx=fwd.index
    if start: idx=idx[idx>=pd.Timestamp(start)]
    ics=[]; ok=0
    for d in idx:
        x=f.loc[d]; y=fwd.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ok+=1; a=x[m].rank().values; b=y[m].rank().values
            if np.std(a)>0 and np.std(b)>0: ics.append(np.corrcoef(a,b)[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return {'IC':0.,'ICIR':0.,'n':len(ics),'hit':0.,'cov':0.,'ok':ok}
    hit=float((ics>0).mean()); cov=float(f.notna().mean().mean())
    mu=ics.mean(); sd=ics.std(); icir=mu/sd*np.sqrt(len(ics)) if sd>0 else 0
    return {'IC':float(mu),'ICIR':float(icir),'n':len(ics),'hit':hit,'cov':cov,'ok':ok}

def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0); return float((s.diff()!=0).mean().mean())

def report(name,fv):
    fv=fv.reindex(fwd10.index)
    a=compute_ic(fv,fwd10); b=compute_ic(fv,fwd5); c=compute_ic(fv,fwd20)
    print(f"{name}: IC={a['IC']:.4f} ICIR={a['ICIR']:.4f} n={a['n']} ok={a['ok']} hit={a['hit']:.3f} cov={a['cov']:.3f} tov={turnover(fv):.3f} | [5]{b['IC']:.4f} [20]{c['IC']:.4f} [20i]{c['ICIR']:.4f}", flush=True)
    return a

def autocorr_panel(df,win):
    out=pd.DataFrame(index=df.index,columns=df.columns,dtype=float)
    for col in df.columns:
        s=df[col]
        out[col]=s.rolling(win,min_periods=win//2).apply(lambda x: np.corrcoef(x[1:],x[:-1])[0,1] if len(x)>2 else np.nan, raw=False)
    return out

print("\n===== REVALIDATE EFFECTIVE LIBRARY (10d horizon) =====")
report("ac1_120d", autocorr_panel(close.pct_change(),120)) if False else report("ac1_120d", rets.rolling(120,min_periods=60).apply(lambda x: np.corrcoef(x[1:],x[:-1])[0,1] if len(x)>3 else np.nan, raw=False))
report("bb_width_20d", (2*rets.rolling(20).std()).div(close))
report("beta_VIX_60", (rets.rolling(60).cov(dVIX)).div(dVIX.rolling(60).var()))
report("cny_beta_60", (rets.rolling(60).cov(dCNY)).div(dCNY.rolling(60).var()))
dxyr=dDXY.reindex(close.index)
report("dxy_corr_change_20_60", rets.rolling(20).corr(dxyr)-rets.rolling(60).corr(dxyr))
report("kaufman_eff_20d", (close.diff(20).abs()).div(close.diff().abs().rolling(20).sum()))
report("kurt_20d", (rets-rets.rolling(20).mean()).pow(4).rolling(20).mean().div(rets.rolling(20).std().pow(4))-3)
report("mom_10d_skip5", close.shift(5)/close.shift(15)-1.0)
report("mom_120d_skip5", close.shift(5)/close.shift(125)-1.0)
report("mom_10_vixreg", (close/close.shift(5)-1)*np.sign(dVIX.diff(10).shift(5)).reindex(close.index))
report("rng_pos_20d", (rets.clip(lower=0).rolling(20).mean()).div(rets.rolling(20).std()))
report("skew_20d", (rets-rets.rolling(20).mean()).pow(3).rolling(20).mean().div(rets.rolling(20).std().pow(3)))
report("vol_z20_volume", (vol - vol.rolling(20).mean())/vol.rolling(20).std())

print("\n===== SWEEP NEW CANDIDATES =====")
report("NEW retrace_high120", close/close.rolling(120).max()-1)
report("NEW vol_ratio_10_60", rets.rolling(10).std()/rets.rolling(60).std())
dd=rets.where(rets<0,0)
report("NEW downside_vol_60", dd.rolling(60).std())
report("NEW autocorr_10d", autocorr_panel(rets, 30))
report("NEW eff_ratio_10", (close.diff(10).abs()).