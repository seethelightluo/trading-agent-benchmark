"""miner3_20350913_exp01_tsq_trendstrength.py
Factor: Trend Strength Quality (tsq_20d / tsq_40d)
Concept: R-squared of log-price linear trend over a rolling window.
Directional moves with high trend consistency (clean, high R2) may persist,
distinct from pure momentum magnitude.
Universe: 15 cross-asset tradable instruments.
Gates: abs(IC)>=0.0070 AND abs(ICIR)>=0.084 (same-horizon, 10d).
"""
from alphacrafter.sim.utils import get_stock_daily_data
import numpy as np
import pandas as pd

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
             'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
N_DAYS = 2400
VISIBLE = pd.Timestamp('2035-09-12')

print("MINER3 EXP01 | TSQ trend strength quality | visible", VISIBLE.date(), flush=True)

inst_data = {}
for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=N_DAYS)
    if df is not None and len(df) > 120:
        s = df.set_index('date')['close'].astype(float)
        s = s[s.index <= VISIBLE]
        inst_data[sym] = s
close = pd.DataFrame(inst_data)
print(f"Close panel: {close.shape[0]} dates x {close.shape[1]} assets | "
      f"{close.index[0].date()}..{close.index[-1].date()}", flush=True)
rets = close.pct_change().dropna()

fwd_5d = rets.shift(-5).rolling(5).sum()
fwd_10d = rets.shift(-10).rolling(10).sum()
fwd_20d = rets.shift(-20).rolling(20).sum()

def compute_tsq(close_panel, w=20):
    out = pd.DataFrame(np.nan, index=close_panel.index, columns=close_panel.columns)
    vals = close_panel.values.astype(float)
    x = np.arange(w, dtype=float)
    xm = x.mean(); xd = x - xm
    ss_xx = (xd**2).sum()
    for i in range(w, len(close_panel)):
        chunk = vals[i-w:i]
        ly = np.log(np.maximum(chunk, 1e-9))
        ys = ly.sum(axis=0)
        sum_xy = (ly.T * x).sum(axis=0)
        b = (sum_xy - w * xm * ys) / ss_xx
        a = (ys - b * w * xm) / w
        y_pred = a + b[None, :] * x[:, None]
        ss_res = ((ly - y_pred)**2).sum(axis=0)
        ss_tot = ((ly - ys/w)**2).sum(axis=0)
        r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res/ss_tot, np.nan)
        out.iloc[i] = r2
    return out

def compute_ic(fv, fwd, min_assets=8, min_dates=30):
    common = sorted(set(fv.index) & set(fwd.index))
    ics = []
    for d in common:
        x = fv.loc[d]; y = fwd.loc[d]
        m = x.notna() & y.notna()
        if m.sum() < min_assets: continue
        xv = x[m].rank().values; yv = y[m].rank().values
        if np.std(xv) > 0 and np.std(yv) > 0:
            ics.append(np.corrcoef(xv, yv)[0,1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return {'IC':0.0,'ICIR':0.0,'n':len(ics),'hit':0.0}
    mu,sd = float(ics.mean()), float(ics.std())
    ir = mu/sd*np.sqrt(len(ics)) if sd>0 else 0.0
    return {'IC':mu,'ICIR':ir,'n':len(ics),'hit':float((ics>0).mean())}

def report(name, fv):
    a5 = compute_ic(fv, fwd_5d); a10 = compute_ic(fv, fwd_10d); a20 = compute_ic(fv, fwd_20d)
    ok = abs(a10['IC'])>=0.0070 and abs(a10['ICIR'])>=0.084
    flag = 'OK' if ok else '--'
    cov = fv.notna().mean().mean()
    tr = fv.rank(axis=1).diff(10).abs().mean().mean()
    print(f"  [{flag}] {name:16s} IC10={a10['IC']:+.4f} ICIR10={a10['ICIR']:+.4f} "
          f"n={a10['n']:4d} hit={a10['hit']:.3f} | IC5={a5['IC']:+.3f} IC20={a20['IC']:+.3f} "
          f"cov={cov:.3f} turn={tr:.3f}", flush=True)
    return a10, ok

print("\n--- Factor results ---", flush=True)
for w in [10, 20, 40, 60]:
    tsq = compute_tsq(close, w=w)
    report(f"tsq_{w}d", tsq)

def compute_tsq_signed(close_panel, w=20):
    tsq = compute_tsq(close_panel, w)
    mom = close_panel/close_panel.shift(w) - 1.0
    return tsq * np.sign(mom)

print("\n--- Signed (trend direction interacted) ---", flush=True)
for w in [20, 40]:
    s = compute_tsq_signed(close, w)
    report(f"tsq_signed_{w}d", s)

print("\nDONE", flush=True)