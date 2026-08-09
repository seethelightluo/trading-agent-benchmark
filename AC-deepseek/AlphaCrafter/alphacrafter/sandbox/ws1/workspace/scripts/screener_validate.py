"""Screener validation: recent IC + factor cross-correlation for the 4 admitted factors."""
import pandas as pd
import numpy as np
import os

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA = "../persistent/stock_data"
IDX = "../persistent/index_data"

def load(sym, folder=DATA):
    df = pd.read_csv(os.path.join(folder, sym + ".csv"), parse_dates=[0])
    df.columns = [c.strip() for c in df.columns]
    dcol = [c for c in df.columns if c.lower() in ("date", "datetime")][0]
    df = df.set_index(pd.to_datetime(df[dcol])).sort_index()
    ccol = [c for c in df.columns if c.lower() == "close"][0]
    return df[ccol]

close = pd.DataFrame({s: load(s) for s in ASSETS})
vix = load("VIX", IDX)
close = close[close.index <= "2026-07-15"]
vix = vix[vix.index <= "2026-07-15"]
rets = close.pct_change()

# ---- factor computations (raw, before direction flip) ----
f_mom10 = close.shift(5) / close.shift(15) - 1.0
f_mom120 = close.shift(5) / close.shift(125) - 1.0
f_vov = rets.rolling(20).std().rolling(60).std()
beta60 = rets.rolling(60).cov(vix.pct_change()) / vix.pct_change().rolling(60).var()
vix_move = vix / vix.shift(20) - 1.0
f_vixb = -beta60.multiply(vix_move, axis=0)

factors = {
    "mom_10d_skip5": f_mom10,
    "mom_120d_skip5": f_mom120,
    "vol_of_vol20x60": f_vov,
    "vix_beta_cond_60x20": f_vixb,
}

# forward 10d returns (matching admission horizon)
fwd = close.shift(-10) / close - 1.0
valid = fwd.notna() & close.notna()

# ---- recent-window IC (last 250 trading days, plus full-warmup for reference) ----
def ic_stats(f, fwd, mask):
    out = []
    for t in mask.index[mask.any(axis=1)]:
        y = fwd.loc[t][mask.loc[t]]
        x = f.loc[t][mask.loc[t]]
        if len(x) >= 8 and x.nunique() > 1 and y.nunique() > 1:
            out.append((t, np.corrcoef(x, y)[0, 1]))
    return out

print("=== Rank-IC by window (10d forward, direction NOT applied) ===")
for name, f in factors.items():
    m = valid & f.notna()
    full = ic_stats(f.rank(axis=1), fwd, m)
    recent = [x for x in full if x[0] >= pd.Timestamp("2025-07-16")]
    last60 = [x for x in full if x[0] >= pd.Timestamp("2026-05-16")]
    def summ(seq):
        if not seq:
            return (np.nan,)*4
        ics = [s[1] for s in seq]
        return (np.mean(ics), np.std(ics), np.mean(ics)/(np.std(ics)+1e-12), len(ics))
    fu, fs, fi, fn = summ(full)
    ru, rs, ri, rn = summ(recent)
    lu, ls, li, ln = summ(last60)
    print(f"{name:20s} FULL ic={fu:+.4f} icir={fi:+.2f} n={fn:4.0f} | "
          f"1Y ic={ru:+.4f} icir={ri:+.2f} n={rn:4.0f} | "
          f"60d ic={lu:+.4f} icir={li:+.2f} n={ln:3.0f}")

# ---- factor cross-correlation (rank corr of latest cross-section) ----
print("\n=== Factor cross-sectional rank correlation (last 60 valid dates, avg) ===")
names = list(factors.keys())
corr_acc = { (i,j): [] for i in range(4) for j in range(i+1,4) }
for t in close.index[-200:]:
    rows = []
    ok = True
    for f in factors.values():
        r = f.loc[t].rank()
        rows.append(r)
        if r.notna().sum() < 8:
            ok = False
    if not ok:
        continue
    X = pd.concat(rows, axis=1)
    X.columns = names
    for i in range(4):
        for j in range(i+1,4):
            c = X[names[i]].corr(X[names[j]])
            if np.isfinite(c):
                corr_acc[(i,j)].append(c)
for (i,j), vals in corr_acc.items():
    if vals:
        print(f"{names[i]:20s} vs {names[j]:20s}: avg rank corr = {np.mean(vals):+.3f} (n={len(vals)})")

# ---- current cross-sectional exposure snapshot (ranked, direction applied) ----
print("\n=== Latest factor snapshot (z-scored, direction applied) ===")
snap = pd.DataFrame(index=ASSETS)
for name, f in factors.items():
    z = (f.iloc[-1] - f.iloc[-1].mean()) / f.iloc[-1].std()
    snap[name + "_raw_z"] = z.round(2)
dirs = {"mom_10d_skip5": 1, "mom_120d_skip5": 1, "vol_of_vol20x60": 1, "vix_beta_cond_60x20": -1}
snap["composite"] = sum(dirs[n] * snap[n + "_raw_z"] for n in names).round(2)
print(snap.sort_values("composite", ascending=False).to_string())
