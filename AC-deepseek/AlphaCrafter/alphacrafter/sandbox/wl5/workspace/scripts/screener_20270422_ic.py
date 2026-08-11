"""SCREENER 2027-04-22: factor IC assessment + regime check for ensemble update."""
import pandas as pd, numpy as np

ASSETS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
FROZEN = {"000300.SH","000688.SH","CN10Y"}
ACTIVE = [s for s in ASSETS if s not in FROZEN]
END = "2027-04-21"

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

def tail_ratio_20():
    # right-tail mass / left-tail mass over 20d
    q = ret.rolling(20).quantile(0.95)
    lq = ret.rolling(20).quantile(0.05)
    return q.abs() / (lq.abs() + 1e-12)

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
    return ics

factors = {
    "trend_r2_30_signed": (trend_r2_30(), 1),
    "semi_down_ratio_20": (semi_down_20(), -1),
    "mom_120d_skip5": (px/px.shift(120)-1, 1),
    "mom_10d_skip5": (px/px.shift(10)-1, 1),
    "vol_of_vol20x60": (vol_of_vol(), 1),
    "kurt_20": (kurt_20(), 1),
    "time_under_water_120": (time_under_water(), -1),
    "tail_ratio_20": (tail_ratio_20(), 1),
    "dxy_beta_60": (dxy_beta(), 1),
    "WTI_BETA_60": (wti_beta(), 1),
    "vix_beta_cond_60x20": (vix_beta(), -1),
}

print("=== factor IC: last 60d / 120d / 240d (12-asset active cross-section) ===")
res = {}
for name, (fval, exp_dir) in factors.items():
    ics = fwd10_ic(fval)
    if len(ics) >= 6:
        ic60 = ics.iloc[-6:].mean(); icir60 = ic60/(ics.iloc[-6:].std()+1e-9)
        ic120 = ics.iloc[-12:].mean(); icir120 = ic120/(ics.iloc[-12:].std()+1e-9)
        ic240 = ics.iloc[-24:].mean(); icir240 = ic240/(ics.iloc[-24:].std()+1e-9)
        consistent = ic120 * exp_dir
        res[name] = dict(ic60=ic60, icir60=icir60, ic120=ic120, icir120=icir120, ic240=ic240, icir240=icir240, exp_dir=exp_dir, consistent=consistent)
        print(f"{name:22s} dir={exp_dir:+d} ic60={ic60:+.4f}({icir60:+.2f}) ic120={ic120:+.4f}({icir120:+.2f}) ic240={ic240:+.4f}({icir240:+.2f}) consistent120={consistent:+.4f} {'OK' if consistent>0 else 'INVERTED'}")

print("\n=== quality_ic_tilt weights (q=|ic120|*|icir120|, preserve sign(ic120)) ===")
rows = []
for name, v in res.items():
    q = abs(v['ic120']) * abs(v['icir120'])
    rows.append((name, v['ic120'], v['icir120'], q, v['exp_dir']))
rows.sort(key=lambda r: -r[3])
total = sum(r[3] for r in rows) + 1e-12
for name, ic, icir, q, d in rows:
    print(f"{name:22s} ic120={ic:+.4f} icir120={icir:+.2f} q={q:.4f} w={q/total:.4f} dir={d:+d}")

print("\n=== regime snapshot ===")
for s in ["SPX","NDX","SOX","HSI","N225","SX5E","XAU","COPPER","WTI","BTC","ETH","US10Y"]:
    c = px[s]
    r20 = c.iloc[-1]/c.iloc[-21]-1 if len(c)>21 else np.nan
    r60 = c.iloc[-1]/c.iloc[-61]-1 if len(c)>61 else np.nan
    r120 = c.iloc[-1]/c.iloc[-121]-1 if len(c)>121 else np.nan
    ma60 = c.rolling(60).mean().iloc[-1]
    print(f"{s:8s} last={c.iloc[-1]:12.2f} r20={r20:+.3f} r60={r60:+.3f} r120={r120:+.3f} vsMA60={c.iloc[-1]/ma60-1:+.3f}")

vix = pd.read_csv("../persistent/index_data/VIX.csv"); vix["date"]=pd.to_datetime(vix["date"])
vix = vix[vix["date"]<=END].set_index("date")["close"].astype(float)
print(f"\nVIX last={vix.iloc[-1]:.2f} mean60={vix.tail(60).mean():.2f} pctile90={vix.tail(250).quantile(0.9):.2f}")

# cross-sectional dispersion: avg pairwise corr of 20d returns over last 60d
r20 = ret.tail(60)
corrs = []
for i in range(len(ASSETS)):
    for j in range(i+1, len(ASSETS)):
        a, b = ASSETS[i], ASSETS[j]
        m = r20[[a,b]].dropna()
        if len(m) >= 20:
            corrs.append(np.corrcoef(m[a], m[b])[0,1])
print(f"avg pairwise corr (20d ret, last 60d): {np.mean(corrs):.3f}")
