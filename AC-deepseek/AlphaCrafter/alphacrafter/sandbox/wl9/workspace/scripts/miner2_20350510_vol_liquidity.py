"""miner_2 2035-05-10: volume/liquidity factor exploration (corrected data loading)."""
import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
VISIBLE = pd.Timestamp('2035-05-09')

uni = {}
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None or len(df) < 300:
        df = get_index_daily_data(symbol=s, days=4000)
    if df is not None and len(df) >= 300:
        df = df.copy(); df['date'] = pd.to_datetime(df['date']); df = df.set_index('date').sort_index()
        df = df[df.index <= VISIBLE]
        uni[s] = df

close = pd.DataFrame({s: uni[s]['close'] for s in uni}).sort_index()
vol = pd.DataFrame({s: uni[s]['volume'] for s in uni}).sort_index()
rets = close.pct_change()
fwd_10 = rets.shift(-10)
print(f"close rows={len(close)} assets={close.shape[1]} range {close.index[0].date()}..{close.index[-1].date()}", flush=True)

vol_assets = [a for a in WATCH if a in vol.columns and (vol[a] > 0).mean() > 0.2]
print("Volume-real assets:", vol_assets, flush=True)

def rank_ic(factor_df, fwd_ret, min_valid=8):
    ics = []
    for dt in factor_df.index:
        if dt not in fwd_ret.index: continue
        f = factor_df.loc[dt].astype(float); r = fwd_ret.loc[dt].astype(float)
        m = f.notna() & r.notna()
        if m.sum() < min_valid: continue
        ic = f[m].rank().corr(r[m].rank())
        if not np.isnan(ic): ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0:
        return {"ic": np.nan, "icir": np.nan, "n": 0, "hit": np.nan}
    icm = ics.mean(); icir = icm/ics.std() if ics.std()>0 else np.nan
    return {"ic": icm, "icir": icir, "n": len(ics), "hit": (ics>0).mean()}

def report(name, fdf, fwd):
    r = rank_ic(fdf, fwd, 8)
    cov = fdf.astype(float).notna().mean().mean() if fdf.notna().any().any() else 0
    ok = abs(r['ic'])>=0.0070 and abs(r['icir'])>=0.084 and not np.isnan(r['icir'])
    print(f"[{'OK' if ok else '--'}] {name:22s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} n={r['n']:5d} hit={r['hit']:.3f} cov={cov:.3f}", flush=True)
    return r, ok

amt = close * vol
amt_z = pd.DataFrame(index=close.index, columns=close.columns)
vmom = pd.DataFrame(index=close.index, columns=close.columns)
amihud = pd.DataFrame(index=close.index, columns=close.columns)
for a in vol_assets:
    s = amt[a]
    amt_z[a] = (s - s.rolling(60).mean()) / s.rolling(60).std().replace(0, np.nan)
    m5 = vol[a].rolling(5).mean(); m60 = vol[a].rolling(60).mean()
    vmom[a] = m5 / m60.replace(0, np.nan)
    rr = rets[a].abs()
    amihud[a] = rr / amt[a].replace(0, np.nan)
amt_z = amt_z.replace([np.inf,-np.inf], np.nan)
vmom = vmom.replace([np.inf,-np.inf], np.nan)
amihud = amihud.replace([np.inf,-np.inf], np.nan)

print("\n=== Volume/Liquidity candidates (h=10) ===", flush=True)
for name, f in [("amt_z_60", amt_z), ("vol_mom_5_60", vmom), ("amihud_illiq", amihud)]:
    report(name, f, fwd_10)
print("done", flush=True)