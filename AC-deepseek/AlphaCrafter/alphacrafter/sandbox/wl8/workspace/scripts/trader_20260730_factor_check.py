"""Trader: verify factor computation conventions vs persisted library panels."""
import json, base64, zlib, io, sys
import numpy as np
import pandas as pd

END = "2026-07-29"
ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI", "N225",
          "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]

def load_close(assets, end=END):
    closes = {}
    for a in assets:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= end]
        closes[a] = df.set_index("date")["close"].astype(float)
    return pd.DataFrame(closes)

def load_index(name, end=END):
    df = pd.read_csv(f"../persistent/index_data/{name}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= end]
    return df.set_index("date")["close"].astype(float)

def lib_panel(fname):
    d = json.load(open(f"factors/{fname}.json"))
    art = d["validation"]["signal_artifact"]
    csv = zlib.decompress(base64.b64decode(art["data"])).decode()
    p = pd.read_csv(io.StringIO(csv), index_col=0)
    p.index = pd.to_datetime(p.index)
    return p

def rolling_beta(y, x, n):
    yv = y.pct_change()
    xv = x.reindex(y.index)
    varx = xv.rolling(n).var()
    cov = yv.rolling(n).cov(xv)
    return cov / varx.replace(0, np.nan)

close = load_close(ASSETS)
vix = load_index("VIX")
vix_r = vix.pct_change()

# mom_10d_skip5
mom = close.shift(5) / close.shift(15) - 1.0

# vix_beta_cond_60x20 variants
vixm = vix / vix.shift(20) - 1.0
b_vix = rolling_beta(close, vix_r, 60)
vb_minus = -b_vix * vixm.reindex(close.index)   # as in library expression
vb_plain = b_vix * vixm.reindex(close.index)    # miner_2 make_cond style

# yield_beta_cond_60x20 (use_diff=True)
us10 = close["US10Y"]
b_rate = rolling_beta(close, us10.diff(), 60)
yb = b_rate * us10.diff(20)

lib_mom = lib_panel("mom_10d_skip5")
lib_vb = lib_panel("vix_beta_cond_60x20")
lib_yb = lib_panel("yield_beta_cond_60x20")

def corr2(a, b):
    common = a.index.intersection(b.index)
    cc = []
    for c in a.columns:
        s = a[c].reindex(common)
        t = b[c].reindex(common)
        m = s.notna() & t.notna()
        if m.sum() > 200:
            cc.append(np.corrcoef(s[m], t[m])[0, 1])
    return float(np.mean(cc)), float(np.max(np.abs(cc))) if cc else np.nan, len(cc)

print("mom  :", corr2(mom, lib_mom))
print("vb_minus vs lib:", corr2(vb_minus, lib_vb))
print("vb_plain vs lib:", corr2(vb_plain, lib_vb))
print("yb vs lib:", corr2(yb, lib_yb))

# rank IC of each factor at H=10 over 2020..2026-07-29
def rank_ic(fdf):
    fwd = close.shift(-10) / close - 1.0
    common = fdf.index.intersection(fwd.index)
    ics = []
    for d in common:
        f = fdf.loc[d].dropna(); r = fwd.loc[d].dropna()
        both = f.index.intersection(r.index)
        if len(both) >= 8:
            ic = f[both].rank().corr(r[both].rank())
            if np.isfinite(ic):
                ics.append(ic)
    a = np.array(ics)
    return float(a.mean()), float(a.mean() / a.std(ddof=1)) if len(a) > 2 else np.nan, len(a)

for name, f in [("mom", mom), ("vb_minus", vb_minus), ("vb_plain", vb_plain), ("yb", yb)]:
    print(name, "rankIC10:", rank_ic(f))

# Latest factor snapshot as of END
print("\n--- latest factor snapshot", END, "---")
for name, f in [("mom", mom), ("vb_minus", vb_minus), ("yb", yb)]:
    print(name)
    print(f.loc[END].round(4).to_string())
