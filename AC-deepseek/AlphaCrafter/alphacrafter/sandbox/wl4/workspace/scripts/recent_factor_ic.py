"""SCREENER: recompute active-factor signals from price data and measure recent
cross-sectional rank IC vs forward-10d returns. Uses data only through the
visible date (2030-02-22). No live-account interaction."""
import json, os, math
import pandas as pd
import numpy as np

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA_DIR = "../persistent/stock_data"
VISIBLE = "2030-02-22"

# ---- build close panel ----
panel = {}
for a in ASSETS:
    p = os.path.join(DATA_DIR, a + ".csv")
    df = pd.read_csv(p, parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)]
    s = df.set_index("date")["close"].astype(float)
    panel[a] = s

px = pd.DataFrame(panel).sort_index()
print("panel dates:", px.index.min().date(), "->", px.index.max().date(), "n=", len(px))
print("last close (2030-02-22):")
print(px.iloc[-1].round(3).to_string())

ret = px.pct_change()
mkt = ret.mean(axis=1)  # equal-weight 15

# ---- factor 1: dn_mkt_beta_60d ----
def dn_mkt_beta(win=60, min_obs=40):
    down = mkt.where(mkt < 0)
    out = {}
    for a in ASSETS:
        x = down.values
        y = ret[a].values
        betas = np.full(len(px), np.nan)
        for i in range(win, len(px)):
            seg_x = x[i-win:i]
            seg_y = y[i-win:i]
            m = ~(np.isnan(seg_x) | np.isnan(seg_y))
            if m.sum() >= min_obs:
                sx, sy = seg_x[m], seg_y[m]
                vx = np.var(sx)
                if vx > 1e-12:
                    betas[i] = np.cov(sx, sy)[0, 1] / vx
        out[a] = pd.Series(betas, index=px.index)
    return pd.DataFrame(out)

# ---- factor 2: rate_beta_cn10y_60d ----
def rate_beta(win=60, min_obs=40):
    cn10y = px["CN10Y"].pct_change()
    out = {}
    for a in ASSETS:
        x = cn10y.values
        y = ret[a].values
        betas = np.full(len(px), np.nan)
        for i in range(win, len(px)):
            seg_x = x[i-win:i]
            seg_y = y[i-win:i]
            m = ~(np.isnan(seg_x) | np.isnan(seg_y))
            if m.sum() >= min_obs:
                sx, sy = seg_x[m], seg_y[m]
                vx = np.var(sx)
                if vx > 1e-12:
                    betas[i] = np.cov(sx, sy)[0, 1] / vx
        out[a] = pd.Series(betas, index=px.index)
    return pd.DataFrame(out)

# ---- factor 3: vol_adj_mom_accel_20x60 ----
def vol_adj_mom_accel(fast=20, slow=60, vwin=20):
    mom_f = px / px.shift(fast) - 1.0
    mom_s = px / px.shift(slow) - 1.0
    vol = ret.rolling(vwin).std()
    return (mom_f - mom_s) / vol

factors = {
    "dn_mkt_beta_60d": dn_mkt_beta(),
    "rate_beta_cn10y_60d": rate_beta(),
    "vol_adj_mom_accel_20x60": vol_adj_mom_accel(),
}

# ---- forward 10d return (panel time axis) ----
fwd = px.shift(-10) / px - 1.0

def rank_ic(sig, fwd_ret, min_valid=8):
    ic = []
    for dt in sig.index:
        s = sig.loc[dt]
        f = fwd_ret.loc[dt]
        m = s.notna() & f.notna()
        if m.sum() >= min_valid:
            ic.append((dt, np.corrcoef(s[m].rank(), f[m].rank())[0, 1]))
    return pd.Series(dict(ic))

print("\n=== RECENT FACTOR IC (through", VISIBLE, ") ===")
for name, sig in factors.items():
    ic = rank_ic(sig, fwd)
    ic = ic.dropna()
    for label, n in [("last 63d", 63), ("last 126d", 126), ("last 252d", 252)]:
        sub = ic.iloc[-n:]
        if len(sub) == 0:
            print(f"{name:24s} {label}: no data"); continue
        m = sub.mean()
        sd = sub.std(ddof=1) if len(sub) > 2 else float("nan")
        icir = m / sd if sd and not math.isnan(sd) and sd > 0 else float("nan")
        hit = (sub > 0).mean()
        print(f"{name:24s} {label}: n={len(sub):4d} IC={m:+.4f} ICIR={icir:+.3f} hit={hit:.2f} lastIC={sub.iloc[-1]:+.4f}")
    # recent cross-sectional exposure (last date)
    last = sig.iloc[-1]
    print(f"  last-date exposures: {last.dropna().round(3).to_dict()}")

# factor correlation (recent, on overlapping valid dates)
print("\n=== FACTOR CROSS-CORRELATION (last 126d, rank of ranks) ===")
mats = {}
for name, sig in factors.items():
    mats[name] = sig.rank(axis=1)
aligned = pd.concat({k: v for k, v in mats.items()}, axis=1)
aligned = aligned.iloc[-126:]
names = list(factors.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = names[i], names[j]
        corrs = []
        for dt in aligned.index:
            s1 = aligned.loc[dt, a].dropna()
            s2 = aligned.loc[dt, b].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) >= 6:
                corrs.append(np.corrcoef(s1[common], s2[common])[0, 1])
        if corrs:
            print(f"{a} vs {b}: mean_corr={np.mean(corrs):+.3f} (n={len(corrs)})")

# turnover proxy: mean abs change of cross-sectional rank between consecutive dates
print("\n=== RANK TURNOVER (10d, last 126d) ===")
for name, sig in factors.items():
    r = sig.rank(axis=1)
    chg = (r - r.shift(10)).abs().mean(axis=1).dropna().iloc[-126:]
    print(f"{name:24s} mean_abs_rank_change/10d = {chg.mean():.3f}")
