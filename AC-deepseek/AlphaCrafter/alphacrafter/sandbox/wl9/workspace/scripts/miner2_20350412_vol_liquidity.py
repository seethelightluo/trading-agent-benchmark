"""miner_2 2035-04-12: volume/liquidity factor exploration.
Fresh candidate - volume data real only for index/crypto assets (not commodities/US10Y).
Goal: find a volume-informed factor that complements existing library (momentum, vol, macro-beta).
Honest reporting of volume coverage so universe note is respected.
"""
import pandas as pd, numpy as np, json

ASSETS = ["000300.SH","000688.SH","SPX","NDX","SOX","HSI","N225","SX5E",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
VISIBLE = "2035-04-11"

def load_panel():
    panel = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df[df.index <= pd.Timestamp(VISIBLE)]
        panel[a] = df
    return panel

def load_index(name):
    df = pd.read_csv(f"../persistent/index_data/{name}.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df.rename(columns={df.columns[1]: 'close'})['close']

panel = load_panel()
# close, volume, amount = close*volume
close = pd.DataFrame({a: p['close'] for a, p in panel.items()}).sort_index()
vol   = pd.DataFrame({a: p['volume'] for a, p in panel.items()}).sort_index()
amount = close * vol
rets = close.pct_change()

# Volume has real data only for a subset. Define a mask of assets with genuine trading volume
# (median volume > 0 and at least 20% non-zero).
vol_assets = [a for a in ASSETS if (vol[a] > 0).mean() > 0.2]
print("Volume-real assets:", vol_assets, flush=True)

# Liquidity factor: turnover per notional (amount) scaled - but amount limited to same subset.
# Build amount z-score factor on vol_assets only.
amt_z = pd.DataFrame(index=close.index, columns=ASSETS)
for a in vol_assets:
    s = amount[a]
    amt_z[a] = (s - s.rolling(60).mean()) / s.rolling(60).std()
# for non-volume assets leave NaN (only volume-informed part is priced on volume trades)

# Volume momentum: recent 5d avg volume / 60d avg volume (short-term liquidity expansion)
vmom = pd.DataFrame(index=close.index, columns=ASSETS)
for a in vol_assets:
    m5 = vol[a].rolling(5).mean(); m60 = vol[a].rolling(60).mean()
    vmom[a] = m5 / m60
vmom = vmom.replace([np.inf, -np.inf], np.nan)

# Coin-specific amount momentum proxy using close*volume (works for all, but zeros for commodities)

# Amihud-like: |ret| / amount  (illiquidity); only on vol_assets
amihud = pd.DataFrame(index=close.index, columns=ASSETS)
for a in vol_assets:
    r = rets[a].abs()
    amihud[a] = r / amount[a].replace(0, np.nan)

def fwd(frame, horizon):
    return pd.DataFrame({a: frame[a].shift(-horizon)/frame[a] - 1.0 for a in frame.columns})

def rank_ic(factor_df, fwd_df, min_valid=8):
    common = factor_df.index.intersection(fwd_df.index)
    ics = []
    for dt in common:
        f, r = factor_df.loc[dt], fwd_df.loc[dt]
        m = f.isna() | r.isna() | np.isinf(f)
        fv, rv = f[~m], r[~m]
        if len(fv) >= min_valid:
            ic = fv.rank().corr(rv.rank())
            if not np.isnan(ic):
                ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0:
        return {"n_ic_dates": 0, "ic": np.nan, "icir": np.nan, "ic_hit": np.nan}
    icm = ics.mean(); icir = icm / ics.std() if ics.std() > 0 else np.nan
    return {"n_ic_dates": len(ics), "ic": icm, "icir": icir, "ic_hit": (ics > 0).mean()}

def report(name, fdf, fwd10):
    r = rank_ic(fdf, fwd10, 8)
    # coverage = fraction of non-null cells across all assets (naive daily panel)
    cov = fdf.notna().mean().mean() if fdf.notna().any().any() else 0
    ok = abs(r['ic']) >= 0.0070 and abs(r['icir']) >= 0.084 and not np.isnan(r['icir'])
    print(f"[{'OK ' if ok else '-- '}] {name:22s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} "
          f"ndates={r['n_ic_dates']:5d} hit={r['ic_hit']:.3f} pop_cov={cov:.3f}", flush=True)
    return r, ok

fwd10 = fwd(close, 10)
print("=== Volume/Liquidity candidate factors (h=10) ===", flush=True)
cands = {"amt_z_60": amt_z, "vol_mom_5_60": vmom, "amihud": amihud}
for name, f in cands.items():
    report(name, f, fwd10)
print("done", flush=True)