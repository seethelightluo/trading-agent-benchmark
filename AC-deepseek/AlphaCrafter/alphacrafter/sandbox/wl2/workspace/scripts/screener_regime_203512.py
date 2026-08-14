"""Screener regime assessment for 2035-12-20 cycle (data through 2035-12-19).
Reads persisted price CSVs only; no backtest/step imports, no rebalance calls."""
import pandas as pd, numpy as np, os, json

BASE = "../persistent/stock_data"
IDX = "../persistent/index_data"
SYMS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def load(sym):
    p = os.path.join(BASE, f"{sym}.csv")
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    date_col = "date" if "date" in df.columns else [c for c in df.columns if "date" in c.lower()][0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    close_col = "close" if "close" in df.columns else [c for c in df.columns if c.lower()=="close"][0]
    return df[close_col].astype(float)

px = {s: load(s) for s in SYMS}
# align to common calendar (use union, forward fill within each series' own index)
alldates = sorted(set().union(*[set(p.index) for p in px.values()]))
PX = pd.DataFrame({s: px[s].reindex(alldates).ffill() for s in SYMS})
PX = PX.dropna(how="all")
last = PX.index[-1]
print("last date:", last.date())

def ret(horizon):
    return PX.iloc[-1] / PX.iloc[-1-horizon] - 1.0

r5, r20, r60, r120, r180 = ret(5), ret(20), ret(60), ret(120), ret(180)

# range position 252
win = PX.tail(252)
rng_min = win.min()
rng_max = win.max()
range_pos = (PX.iloc[-1] - rng_min) / (rng_max - rng_min).replace(0, np.nan)

# 20d realized vol (ann)
r = PX.pct_change()
vol20 = r.tail(20).std() * np.sqrt(252)

# distance off 252d high
off_high = PX.iloc[-1] / win.max() - 1.0

tab = pd.DataFrame({
    "close": PX.iloc[-1], "r5": r5, "r20": r20, "r60": r60, "r120": r120,
    "r180": r180, "range_pos252": range_pos, "off_high252": off_high, "vol20_ann": vol20})
print(tab.round(4).to_string())

# --- macro observation signals ---
vix = pd.read_csv(os.path.join(IDX, "VIX.csv"))
vix.columns = [c.strip() for c in vix.columns]
vc = [c for c in vix.columns if c.lower() in ("close","value","price")][0]
vix_dates = pd.to_datetime(vix.iloc[:, 0])
vix_s = pd.Series(vix[vc].astype(float).values, index=vix_dates).sort_index()
vix_s = vix_s[~vix_s.index.duplicated(keep="last")]
print("\nVIX last:", vix_s.iloc[-1], "r20:", vix_s.iloc[-1]/vix_s.iloc[-21]-1, "mean60:", vix_s.tail(60).mean())

for sym in ["DXY","USDJPY","EURUSD","USDCNY"]:
    try:
        d = pd.read_csv(os.path.join(IDX, f"{sym}.csv"))
        d.columns = [c.strip() for c in d.columns]
        cc = [c for c in d.columns if c.lower() in ("close","value","price")][0]
        dt = pd.to_datetime(d.iloc[:, 0])
        s = pd.Series(d[cc].astype(float).values, index=dt).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        print(f"{sym}: last {s.iloc[-1]:.4f} r20 {s.iloc[-1]/s.iloc[-21]-1:+.4f} r60 {s.iloc[-1]/s.iloc[-61]-1:+.4f}")
    except Exception as e:
        print(sym, "err", e)

# --- pairwise 60d correlation of tradable returns ---
corr60 = r.tail(60).corr()
vals = corr60.values[np.triu_indices(15, k=1)]
print("\nmean pairwise 60d corr:", vals.mean().round(4), "median:", np.median(vals).round(4))
print("min/max pair corr:", vals.min().round(3), vals.max().round(3))

# --- cross-sectional dispersion ---
print("\n20d spread (pp):", ((r20.max()-r20.min())*100).round(1), "| 60d spread:", ((r60.max()-r60.min())*100).round(1))
print("r20 top3:", r20.nlargest(3).round(4).to_dict())
print("r20 bot3:", r20.nsmallest(3).round(4).to_dict())

# --- SPX corr factor, downbeta, max consec gain, mom180, range_pos (raw cross-section) ---
spx_ret = r["SPX"]
def downbeta(y):
    m = np.polyfit(spx_ret.reindex(y.index).values, y.values, 1)
    return m[0]
downbeta_60 = r["SPX"].rolling(60, min_periods=20).corr(r) * (r["SPX"].rolling(60).std()/r.rolling(60).std())
# simpler: regress each asset's 60d returns on spx returns for negative-spx days
res = {}
for s in SYMS:
    rr = pd.concat([r[s], spx_ret], axis=1, keys=["a","spx"]).dropna().tail(60)
    neg = rr[rr["spx"] < 0]
    if len(neg) >= 15:
        b = np.polyfit(neg["spx"], neg["a"], 1)[0]
    else:
        b = np.nan
    res[s] = b
print("\ndownbeta60 (neg-spx days):", {k: round(v,3) for k,v in res.items()})

# max consecutive gain 20
def longest_run(x):
    best = cur = 0
    for v in x:
        if v > 0: cur += 1; best = max(best, cur)
        else: cur = 0
    return best
mcg = {}
for s in SYMS:
    mcg[s] = r[s].tail(20).rolling(21, min_periods=10).apply(longest_run, raw=True).iloc[-1]
print("\nmax_consec_gain_20:", {k: round(v,2) for k,v in mcg.items()})

# mom_180d_skip5 raw
mom180 = PX.iloc[-1].shift(0)  # placeholder
mom180_raw = {s: (PX[s].iloc[-1-5]/PX[s].iloc[-1-185]-1) if len(PX) > 190 else np.nan for s in SYMS}
print("\nmom_180d_skip5:", {k: round(v,4) for k,v in mom180_raw.items()})

# spx_corr60
spx_corr60 = r.tail(60).corr()["SPX"]
print("\nspx_corr60:", {k: round(v,3) for k,v in spx_corr60.items()})
