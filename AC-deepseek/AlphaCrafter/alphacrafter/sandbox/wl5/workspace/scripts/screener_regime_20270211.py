"""SCREENER: regime assessment + recent factor IC check as of 2027-02-10."""
import pandas as pd, numpy as np, json, glob, os

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
END = "2027-02-10"

def load(sym):
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= END].set_index("date").sort_index()
    return df["close"].astype(float)

px = pd.DataFrame({s: load(s) for s in ASSETS}).dropna(how="all")
px = px.ffill().dropna()
print("price panel:", px.shape, px.index.min().date(), "->", px.index.max().date())

ret = px.pct_change()
last = px.index[-1]

# --- trend / regime stats (last 250d) ---
win = px.loc[px.index[-250]:]
ma20 = win.mean(axis=1).rolling(20).mean().dropna()
ma60 = win.mean(axis=1).rolling(60).mean().dropna()
px_avg = win.mean(axis=1)
print("\n=== cross-asset average (equal-weight, 15 assets) ===")
print(f"last close avg: {px_avg.iloc[-1]:.1f}, MA20: {ma20.iloc[-1]:.1f}, MA60: {ma60.iloc[-1]:.1f}")
print(f"avg vs MA20: {px_avg.iloc[-1]/ma20.iloc[-1]-1:+.3%}, avg vs MA60: {px_avg.iloc[-1]/ma60.iloc[-1]-1:+.3%}")
print(f"20d ret: {px_avg.iloc[-1]/px_avg.iloc[-21]-1:+.3%}, 60d ret: {px_avg.iloc[-1]/px_avg.iloc[-61]-1:+.3%}, 120d ret: {px_avg.iloc[-1]/px_avg.iloc[-121]-1:+.3%}")

# per-asset recent returns
print("\n=== per-asset 20/60/120d returns ===")
for s in ASSETS:
    c = px[s]
    r20 = c.iloc[-1]/c.iloc[-21]-1 if len(c)>21 else np.nan
    r60 = c.iloc[-1]/c.iloc[-61]-1 if len(c)>61 else np.nan
    r120 = c.iloc[-1]/c.iloc[-121]-1 if len(c)>121 else np.nan
    print(f"{s:10s} r20 {r20:+8.2%}  r60 {r60:+8.2%}  r120 {r120:+8.2%}")

# breadth
above20 = (px.iloc[-1] > px.rolling(20).mean().iloc[-1]).mean()
above60 = (px.iloc[-1] > px.rolling(60).mean().iloc[-1]).mean()
print(f"\nbreadth above MA20: {above20:.0%}, above MA60: {above60:.0%}")

# volatility regime
vol20 = ret.rolling(20).std().mean(axis=1) * np.sqrt(252)
vol60 = ret.rolling(60).std().mean(axis=1) * np.sqrt(252)
print(f"avg ann. vol (20d): {vol20.iloc[-1]:.1%}, (60d): {vol60.iloc[-1]:.1%}, vol20 6m ago: {vol20.iloc[-126]:.1%}")

# dispersion (cross-sectional std of 20d returns)
disp20 = ret.rolling(20).sum().std(axis=1)
print(f"cross-sectional dispersion (20d ret std): {disp20.iloc[-1]:.3f}, 6m ago: {disp20.iloc[-126]:.3f}")

# correlation regime: avg pairwise corr of 60d returns
r60w = ret.tail(60)
corr = r60w.corr()
avg_corr = (corr.values[np.triu_indices_from(corr.values, k=1)]).mean()
print(f"avg pairwise 60d corr: {avg_corr:.3f}")

# max drawdown from 120d high
dd = px / px.rolling(120).max() - 1
print(f"avg drawdown from 120d high: {dd.iloc[-1].mean():.1%}, min: {dd.iloc[-1].min():.1%}")

# macro obs (VIX from index_data)
vix = pd.read_csv("../persistent/index_data/VIX.csv")
vix["date"] = pd.to_datetime(vix["date"])
vix = vix[vix["date"] <= END].set_index("date")["close"].astype(float)
print(f"\nVIX last: {vix.iloc[-1]:.2f}, 1m ago: {vix.iloc[-22]:.2f}, 3m ago: {vix.iloc[-63]:.2f}")

# --- recent factor IC check ---
print("\n=== recent 120d factor IC (rank IC vs 10d fwd ret, 10d overlap) ===")
fwd = ret.shift(-10)
def rank_ic_series(fval, fwdret):
    ics = []
    for t in range(60, len(fval)-10, 10):
        x = fval.iloc[t]
        y = fwdret.shift(10).iloc[t]  # forward 10d return realized
        m = x.notna() & y.notna()
        if m.sum() >= 8:
            ics.append(np.corrcoef(x[m].rank(), y[m].rank())[0,1])
    return np.array(ics)

# load active factor json metadata + recompute simple versions where possible
import sys
sys.path.insert(0, ".")

def fwd10_ic(factor_series):
    s = factor_series.reindex(px.index)
    ic_series = []
    for i in range(0, len(s)-10, 10):
        x = s.iloc[i]
        y = ret.iloc[i+10]
        m = x.notna() & y.notna()
        if m.sum() >= 8:
            ic_series.append(np.corrcoef(x[m].rank(), y[m].rank())[0,1])
    return ic_series

# recompute factor values from price data
logp = np.log(px)
def trend_r2_30():
    out = {}
    for s in ASSETS:
        c = logp[s]
        r = {}
        for t in range(30, len(c)):
            y = c.iloc[t-29:t+1].values
            x = np.arange(30)
            cov = np.cov(x, y)[0,1]
            r[c.index[t]] = np.sign(cov) * (cov**2) / (np.var(x)*np.var(y) + 1e-12)
        out[s] = pd.Series(r)
    return pd.DataFrame(out)

def mom_120():
    return px / px.shift(120) - 1

def mom_10():
    return px / px.shift(10) - 1

def semi_down_20():
    neg = ret.clip(upper=0)
    sd = np.sqrt((neg**2).rolling(20).mean())
    tot = ret.rolling(20).std()
    return sd / tot

def kurt_20():
    return ret.rolling(20).kurt()

def vol_of_vol():
    v = ret.rolling(20).std()
    return v.rolling(60).std() / (v.rolling(60).mean()+1e-12)

def time_under_water():
    dd = px / px.rolling(120).max() - 1
    return dd.rolling(120).apply(lambda x: (x < -0.01).sum(), raw=True)

def dxy_beta():
    dxy = pd.read_csv("../persistent/index_data/DXY.csv")
    dxy["date"] = pd.to_datetime(dxy["date"])
    dxy = dxy[dxy["date"] <= END].set_index("date")["close"].astype(float)
    dret = dxy.pct_change().reindex(px.index).ffill()
    out = {}
    for s in ASSETS:
        cov = ret[s].rolling(60).cov(dret)
        var = dret.rolling(60).var()
        out[s] = cov/var
    return pd.DataFrame(out)

def wti_beta():
    wti = px["WTI"]
    wret = wti.pct_change()
    out = {}
    for s in ASSETS:
        cov = ret[s].rolling(60).cov(wret)
        var = wret.rolling(60).var()
        out[s] = cov/var
    return pd.DataFrame(out)

def vix_beta():
    vix = pd.read_csv("../persistent/index_data/VIX.csv")
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix[vix["date"] <= END].set_index("date")["close"].astype(float)
    vret = vix.pct_change().reindex(px.index).ffill()
    out = {}
    for s in ASSETS:
        cov = ret[s].rolling(60).cov(vret)
        var = vret.rolling(60).var()
        out[s] = cov/var
    return pd.DataFrame(out)

factors = {
    "trend_r2_30_signed": (trend_r2_30(), 1),
    "semi_down_ratio_20": (semi_down_20(), -1),
    "mom_120d_skip5": (mom_120(), 1),
    "mom_10d_skip5": (mom_10(), 1),
    "vol_of_vol20x60": (vol_of_vol(), 1),
    "kurt_20": (kurt_20(), 1),
    "time_under_water_120": (time_under_water(), -1),
    "dxy_beta_60": (dxy_beta(), 1),
    "WTI_BETA_60": (wti_beta(), 1),
    "vix_beta_cond_60x20": (vix_beta(), -1),
}

for name, (fval, exp_dir) in factors.items():
    ics = fwd10_ic(fval)
    if len(ics) >= 5:
        ics = np.array(ics)
        ic120 = ics[-12:]
        print(f"{name:22s} dir={exp_dir:+d}  ic120={ic120.mean():+.4f} (icir={ic120.mean()/(ic120.std()+1e-9):+.2f}, n={len(ic120)})  ic_all={ics.mean():+.4f}")

# 2026-08-11 ensemble: trend_r2 .241, semi_down .1852-, mom_120d .1765, vol_of_vol .1254, dxy_beta .1094, time_under_water .0996-, kurt_20 .0629
print("\n=== ensemble v7 quality (admission) ===")
ens = {
    "trend_r2_30_signed": (0.0562, 0.1672),
    "semi_down_ratio_20": (-0.0857, -0.2402),
    "mom_120d_skip5": (0.0521, 0.1381),
    "vol_of_vol20x60": (0.0424, 0.1206),
    "dxy_beta_60": (0.0843, 0.2510),
    "time_under_water_120": (-0.0570, -0.1700),
    "kurt_20": (0.0496, 0.1442),
}
for k,(ic,icir) in ens.items():
    print(f"{k:22s} ic={ic:+.4f} icir={icir:+.3f} q={abs(ic)*abs(icir):.5f}")
