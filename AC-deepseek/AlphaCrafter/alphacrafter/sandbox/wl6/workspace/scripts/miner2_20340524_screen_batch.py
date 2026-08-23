"""miner_2 screen batch (2034-05-24 VIS), current date. Gate: |IC|>=0.0070 and |ICIR|>=0.0840 @ horizon 10."""
import sys
sys.path.insert(0, 'scripts')
from factor_validation_lib import rank_ic_series, align_fwd_returns, load_macro, load_panel, TRADABLE, library_corr
import pandas as pd, numpy as np, math, json, glob

VIS = "2034-05-24"
closes = {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(VIS)].sort_values("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    closes[sym] = df.set_index("date")["close"]
px = pd.DataFrame(closes).ffill().dropna(how="all")
px = px.dropna(axis=1, how="all")
ret = px.pct_change()
print("panel shape:", px.shape, "n assets:", px.shape[1], "n dates:", px.shape[0])

def evalc(f, label):
    if not isinstance(f, pd.DataFrame): print(f"[{label}] NOT DF"); return
    f = f.reindex(px.index)
    ic = rank_ic_series(f, align_fwd_returns(px, 10))
    if len(ic) == 0:
        print(f"[{label}] NO IC DATES"); return None
    icm = float(ic.mean()); icstd = float(ic.std(ddof=1)) if len(ic)>1 else np.nan
    icir = icm/icstd if icstd and math.isfinite(icstd) and icstd>0 else np.nan
    hit = float((ic>0).mean())
    recent = ic[ic.index >= "2032-05-25"]
    ricm = float(recent.mean()) if len(recent) else np.nan
    ricir = ricm/recent.std(ddof=1) if len(recent)>2 and recent.std(ddof=1)>0 else np.nan
    cov = float(f.notna().mean().mean())
    gate = (abs(icm) >= 0.0070) and (abs(icir) >= 0.0840)
    tf = turnover(f)
    print(f"[{label}] n_ic={len(ic)} IC={icm:+.4f} ICIR={icir:+.4f} hit={hit:.3f} "
          f"recent2y_IC={ricm:+.4f} recent_ICIR={ricir:+.4f} cov={cov:.3f} turn={tf:.3f} GATE={'PASS' if gate else 'fail'}")
    return {"label":label,"n_ic":int(len(ic)),"ic":icm,"icir":icir,"hit":hit,
            "recent_ic":ricm,"recent_icir":ricir,"cov":cov,"turn":tf,"gate":gate}

def turnover(f):
    r = f.rank(axis=1, pct=True)
    dr = r.diff().abs()
    return float(dr.mean(axis=1).mean()*10) if len(dr) else np.nan

cands = {}
spx = px["SPX"]

# A. 60d vol trend ratio (recent vol / longer vol) - short vol term structure mean-reversion
cands["vol_term_ratio_20x120"] = ret.rolling(20).std() / ret.rolling(120).std().replace(0,np.nan)

# B. Idiosyncratic (residual) 60d momentum after removing SPX beta
beta = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for a in px.columns:
    beta[a] = ret[a].rolling(60).cov(ret["SPX"]) / ret["SPX"].rolling(60).var().replace(0,np.nan)
resid = px.copy()
for a in px.columns:
    resid[a] = ret[a] - beta[a]*ret["SPX"]
cands["idio_mom60_skip5"] = resid.rolling(60).sum() - resid.shift(5).rolling(60).sum()

# C. Upside/downside return asymmetry (realized skewness) 60d
def sign_ratio(x):
    ups = x.rolling(60).apply(lambda w: w[w>0].sum(), raw=True)
    dns = x.rolling(60).apply(lambda w: abs(w[w<0].sum()), raw=True)
    return ups/(dns.replace(0,np.nan))
cands["updn_ratio_60d"] = pd.DataFrame({a: sign_ratio(ret[a]) for a in px.columns})

# D. Cross-asset dispersion persistence: recent cross-sectional vol / long period
cs_vol = ret.std(axis=1)
cands["cs_disp_ratio"] = cs_vol.rolling(20).mean() / cs_vol.rolling(120).mean().replace(0,np.nan)

# E. VIX-level gated momentum: momentum scaled by 1/VIX regime (low vol => trend) - macro interaction
vix = load_macro("VIX", VIS)
vix_inv = (1.0/vix).rolling(5).mean()
mom60 = px.pct_change(60)
cands["mom60_x_invvix"] = mom60.mul(vix_inv, axis=0).reindex(px.index)

# F. Price distance above/below 20d high (breakout distance, near-high bullish)
cands["dist_20d_high"] = (px / px.rolling(20).max() - 1.0)

# G. 5d return / 20d return coherence (short-term trend confirmation vs long)
cands["mom5_over_mom20"] = px.pct_change(5) / px.pct_change(20).replace(0,np.nan)

# H. Downside deviation ratio (semi-vol as share of total vol) - low-downside vol favored
def down_vol_share(x):
    dv = x[x<0].rolling(120).std()
    return 1.0 - dv/(x.rolling(120).std().replace(0,np.nan))
cands["down_vol_share_120"] = pd.DataFrame({a: down_vol_share(ret[a]) for a in px.columns})

results = []
for name, f in cands.items():
    r = evalc(f, name)
    if r: results.append(r)

# library correlation for passing candidates
print("\n=== LIBRARY CORR (passing candidates only) ===")
library = {}
for p in sorted(glob.glob("factors/*.json")):
    fid = p.split("/")[-1][:-5]
    try:
        d = json.load(open(p))
        params = d.get("parameters", {})
        # reconstruct signal if expressible as formula from close
        sig = None
    except Exception:
        pass
for r in results:
    if r["gate"]:
        print(f"{r['label']}: needs manual signal persist; library_corr computed in persist step")