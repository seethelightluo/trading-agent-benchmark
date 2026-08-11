"""SCREENER: clean factor IC excluding frozen assets + candidate factor correlation."""
import pandas as pd, numpy as np

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
FROZEN = {"000300.SH","000688.SH","CN10Y"}
ACTIVE = [s for s in ASSETS if s not in FROZEN]
END = "2027-02-10"

def load(sym):
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= END].set_index("date").sort_index()
    return df["close"].astype(float)

px = pd.DataFrame({s: load(s) for s in ASSETS}).ffill().dropna()
ret = px.pct_change()
logp = np.log(px)

def trend_r2_30():
    out = {}
    for s in ACTIVE:
        c = logp[s]
        r = {}
        for t in range(30, len(c)):
            y = c.iloc[t-29:t+1].values
            x = np.arange(30)
            cov = np.cov(x, y)[0,1]
            r[c.index[t]] = np.sign(cov) * (cov**2) / (np.var(x)*np.var(y) + 1e-12)
        out[s] = pd.Series(r)
    return pd.DataFrame(out)

def semi_down_20():
    neg = ret.clip(upper=0)
    sd = np.sqrt((neg**2).rolling(20).mean())
    return sd / ret.rolling(20).std()

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
    for s in ACTIVE:
        out[s] = ret[s].rolling(60).cov(dret) / (dret.rolling(60).var()+1e-12)
    return pd.DataFrame(out)

def wti_beta():
    wret = px["WTI"].pct_change()
    out = {}
    for s in ACTIVE:
        out[s] = ret[s].rolling(60).cov(wret) / (wret.rolling(60).var()+1e-12)
    return pd.DataFrame(out)

def vix_beta():
    vix = pd.read_csv("../persistent/index_data/VIX.csv")
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix[vix["date"] <= END].set_index("date")["close"].astype(float)
    vret = vix.pct_change().reindex(px.index).ffill()
    out = {}
    for s in ACTIVE:
        out[s] = ret[s].rolling(60).cov(vret) / (vret.rolling(60).var()+1e-12)
    return pd.DataFrame(out)

fwd = ret.shift(-10)[ACTIVE]

def fwd10_ic(fval, window=120):
    s = fval[ACTIVE].reindex(px.index)
    ics = []
    for i in range(0, len(s)-10, 10):
        x = s.iloc[i]; y = fwd.iloc[i]
        m = x.notna() & y.notna()
        if m.sum() >= 8:
            ics.append((s.index[i], np.corrcoef(x[m].rank(), y[m].rank())[0,1]))
    ics = pd.Series(dict(ics))
    return ics.iloc[-12:] if len(ics) >= 12 else ics

factors = {
    "trend_r2_30_signed": (trend_r2_30(), 1),
    "semi_down_ratio_20": (semi_down_20(), -1),
    "mom_120d_skip5": (px/px.shift(120)-1, 1),
    "mom_10d_skip5": (px/px.shift(10)-1, 1),
    "vol_of_vol20x60": (vol_of_vol(), 1),
    "kurt_20": (kurt_20(), 1),
    "time_under_water_120": (time_under_water(), -1),
    "dxy_beta_60": (dxy_beta(), 1),
    "WTI_BETA_60": (wti_beta(), 1),
    "vix_beta_cond_60x20": (vix_beta(), -1),
}

print("=== recent 120d factor IC (12-asset active cross-section, excluding frozen) ===")
for name, (fval, exp_dir) in factors.items():
    ics = fwd10_ic(fval)
    if len(ics) >= 6:
        ic = ics.mean(); icir = ic/(ics.std()+1e-9)
        consistent = ic * exp_dir
        print(f"{name:22s} dir={exp_dir:+d} ic120={ic:+.4f} icir={icir:+.2f} consistent={consistent:+.4f} {'OK' if consistent>0 else 'INVERTED'}")

print("\n=== candidate factor pairwise corr (last 120d, cross-sectional) ===")
cands = ["dxy_beta_60","WTI_BETA_60","vix_beta_cond_60x20","time_under_water_120","trend_r2_30_signed","kurt_20"]
vals = {}
for name in cands:
    fval, _ = factors[name]
    # cross-sectional vector at last date, plus rolling corr of cross-sectional means
    s = fval[ACTIVE]
    # use cross-sectional rank time series: average of pairwise spearman across dates
    vals[name] = s

# compute pairwise spearman of daily cross-sectional factor vectors over last 120d
from scipy.stats import spearmanr
names = list(vals.keys())
corrs = {}
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = vals[names[i]].tail(120), vals[names[j]].tail(120)
        rho_list = []
        for t in range(len(a)):
            x, y = a.iloc[t], b.iloc[t]
            m = x.notna() & y.notna()
            if m.sum() >= 8:
                rho_list.append(spearmanr(x[m], y[m]).statistic)
        corrs[f"{names[i]}~{names[j]}"] = (np.nanmean(rho_list), len(rho_list))
for k, (rho, n) in sorted(corrs.items(), key=lambda kv: -abs(kv[1][0])):
    print(f"{k:55s} avg_rho={rho:+.3f} n={n}")

# factor value dispersion at last date
print("\n=== last-date factor values (rank across 12 active assets) ===")
for name in cands:
    s = vals[name].iloc[-1].dropna()
    if len(s) >= 8:
        print(f"{name:22s} n={len(s)} min={s.min():+.3f} med={s.median():+.3f} max={s.max():+.3f}")
