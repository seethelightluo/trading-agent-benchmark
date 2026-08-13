"""Screener cycle 2034-09-08: fresh factor panel + IC + regime assessment.
Uses data strictly through visible_through=2034-09-07. No future data."""
import numpy as np
import pandas as pd

ASOF = "2034-09-07"
ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU",
          "COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
WATCH = ASSETS  # 15 tradable

def load(sym):
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= ASOF].reset_index(drop=True)
    df = df.set_index("date")
    return df

# ---- build aligned panel ----
closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
for a in WATCH:
    d = load(a)
    closes[a] = d["close"]; opens[a] = d["open"]; highs[a] = d["high"]
    lows[a] = d["low"]; vols[a] = d["volume"]

px = pd.DataFrame(closes).sort_index()
op = pd.DataFrame(opens).sort_index()
hi = pd.DataFrame(highs).sort_index()
lo = pd.DataFrame(lows).sort_index()
vo = pd.DataFrame(vols).sort_index()
ret = px.pct_change()
lnret = np.log(px).diff()

# VIX for macro factor
vix = pd.read_csv("../persistent/index_data/VIX.csv", parse_dates=["date"])
vix = vix[vix["date"] <= ASOF].set_index("date")
vix_col = vix["close"] if "close" in vix.columns else vix.iloc[:, 1]
vix_ret = vix_col.pct_change()
# align vix to px index
vix_ret = vix_ret.reindex(px.index).ffill()
vix_col = vix_col.reindex(px.index).ffill()

F = {}
F["miner2_20260715_id_rev_1d"] = -(px / op - 1.0)
F["miner2_20260715_nbody_1d"] = -(px - op) / (hi - lo).replace(0, np.nan)
for nd in [1, 2, 3, 5]:
    F[f"miner2_20260715_nclv_{nd}d"] = -(px - lo.rolling(nd).min()) / (hi.rolling(nd).max() - lo.rolling(nd).min()).replace(0, np.nan)
    F[f"miner2_20260715_rev_{nd}d"] = -(lnret.rolling(nd).sum())
F["miner2_20260715_rev_1d_vs"] = -(lnret.diff(1)) / (ret.rolling(20).std().replace(0, np.nan))
F["mom_120d_skip5"] = px.shift(5) / px.shift(125) - 1.0
F["vol_of_vol20x60"] = ret.rolling(20).std().rolling(60).std()
# vix beta conditional
beta = ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var().replace(0, np.nan)
F["vix_beta_cond_60x20"] = -beta * (vix_col / vix_col.shift(20) - 1.0)

# ---- regime metrics ----
print("=" * 80)
print("REGIME ASSESSMENT asof", ASOF)
print("=" * 80)
eqw = px.pct_change().mean(axis=1)
for w in [20, 60, 120]:
    cum = (1 + eqw.iloc[-w:]).prod() - 1
    mean_d = eqw.iloc[-w:].mean()
    ann_vol = eqw.iloc[-w:].std() * np.sqrt(252)
    print(f"{w}d: eqw cum {cum:+.2%}  mean daily {mean_d:+.4%}  ann vol {ann_vol:.1%}")

ma20 = px.rolling(20).mean(); ma60 = px.rolling(60).mean()
print("breadth >MA20:", int((px.iloc[-1] > ma20.iloc[-1]).sum()), "/15",
      " >MA60:", int((px.iloc[-1] > ma60.iloc[-1]).sum()), "/15")
# 20d dispersion
disp20 = ret.iloc[-20:].std(axis=1).mean()
disp60 = ret.iloc[-60:].std(axis=1).mean()
print(f"20d cross-sectional daily dispersion {disp20:.4f}  60d {disp60:.4f}")
m20_vol = ret.iloc[-20:].std().mean() * np.sqrt(252)
print(f"mean 20d ann vol {m20_vol:.1%}")
per_asset_vol = ret.iloc[-20:].std() * np.sqrt(252)
print("per-asset 20d ann vol:")
print(per_asset_vol.sort_values(ascending=False).round(3).to_string())

c20 = (px.iloc[-1] / px.iloc[-21] - 1).sort_values(ascending=False)
c60 = (px.iloc[-1] / px.iloc[-61] - 1).sort_values(ascending=False)
c120 = (px.iloc[-1] / px.iloc[-121] - 1).sort_values(ascending=False)
print("\n20d cumulative returns:")
print(c20.round(4).to_string())
print("\n60d cumulative returns:")
print(c60.round(4).to_string())
print("\n120d cumulative returns:")
print(c120.round(4).to_string())

print("\nVIX last:", round(float(vix_col.iloc[-1]), 2),
      " 10d ago:", round(float(vix_col.iloc[-11]), 2),
      " 20d ago:", round(float(vix_col.iloc[-21]), 2),
      " 60d ago:", round(float(vix_col.iloc[-61]), 2))

# macro observation signals
for m in ["DXY", "USDCNY", "USDJPY", "EURUSD"]:
    try:
        d = pd.read_csv(f"../persistent/index_data/{m}.csv", parse_dates=["date"])
        d = d[d["date"] <= ASOF].set_index("date")
        c = d["close"] if "close" in d.columns else d.iloc[:, 1]
        print(f"{m}: last {float(c.iloc[-1]):.3f}  20d {float(c.iloc[-1]/c.iloc[-21]-1)*100:+.2f}%")
    except Exception as e:
        print(m, "ERR", e)

# ---- fresh IC computation ----
print("\n" + "=" * 80)
print("FRESH IC (60-sample rolling window, dates = last 60 obs)")
print("=" * 80)
fwd = {}
for h in [1, 5, 10]:
    fwd[h] = px.shift(-h) / px - 1.0

def rank_ic(sig, fwd_ret, idx):
    s = sig.loc[idx]; f = fwd_ret.loc[idx]
    ics = []
    for t in idx:
        a = s.loc[t].dropna(); b = f.loc[t].reindex(a.index)
        m = pd.concat([a, b], axis=1).dropna()
        if len(m) < 5:
            continue
        ics.append(m.iloc[:, 0].rank().corr(m.iloc[:, 1].rank()))
    return np.array(ics)

idx_all = px.index
n = len(idx_all)
# use last 120 obs to compute, report on last 60 obs for fresh + full window
results = {}
for name, sig in F.items():
    row = {}
    for h in [1, 5, 10]:
        ics = rank_ic(sig, fwd[h], idx_all[-120:])
        if len(ics) == 0:
            row[f"ic{h}"] = np.nan; row[f"icir{h}"] = 0.0; row[f"hit{h}"] = np.nan; row[f"n{h}"] = 0
            continue
        ic = float(np.nanmean(ics)); icir = float(np.nanmean(ics) / (np.nanstd(ics) + 1e-12))
        # fresh 60-window
        ics60 = ics[-60:]
        ic60 = float(np.nanmean(ics60)); icir60 = float(np.nanmean(ics60) / (np.nanstd(ics60) + 1e-12))
        hit60 = float(np.mean(np.sign(ics60) == np.sign(np.nanmean(ics60)))) if np.nanmean(ics60) != 0 else np.nan
        row[f"ic{h}"] = ic60; row[f"icir{h}"] = icir60; row[f"hit{h}"] = hit60
        row[f"n{h}"] = len(ics60)
    results[name] = row

dfr = pd.DataFrame(results).T
dfr["q10"] = dfr["ic10"].abs() * dfr["icir10"].abs()
dfr["q1"] = dfr["ic1"].abs() * dfr["icir1"].abs()
dfr["q5"] = dfr["ic5"].abs() * dfr["icir5"].abs()
print(dfr.round(4).to_string())

# coverage of vix_beta in fresh window
vb = F["vix_beta_cond_60x20"].iloc[-60:]
print("\nvix_beta fresh-window non-NaN coverage:", float(vb.notna().sum().sum()) / (60 * 15))

# ---- factor correlation (recent 60d, cross-sectional demeaned) ----
print("\n" + "=" * 80)
print("FACTOR PAIRWISE CORR (60d avg of cross-sectional corr)")
print("=" * 80)
names = list(F.keys())
sig_flat = {k: v.iloc[-60:] for k, v in F.items()}
corr_mat = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        cs = []
        for t in sig_flat[a].index:
            x = sig_flat[a].loc[t]; y = sig_flat[b].loc[t]
            m = pd.concat([x, y], axis=1).dropna()
            if len(m) < 5:
                continue
            cs.append(m.iloc[:, 0].corr(m.iloc[:, 1]))
        corr_mat.loc[a, b] = np.nanmean(cs) if cs else np.nan
print(corr_mat.round(2).to_string())

dfr.to_pickle("_screener_fresh_20340908.pkl")
print("\nsaved _screener_fresh_20340908.pkl")
