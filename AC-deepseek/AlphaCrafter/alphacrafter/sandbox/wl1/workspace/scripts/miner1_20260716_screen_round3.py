"""miner_1 screening round 3: novel families orthogonal to reversal/clv/nclv/pos_freq/rs/beta.
Candidates: skew, kurtosis, autocorr, efficiency ratio, slope t-stat, R2, downside/upside vol,
gap (overnight), volume-price corr, vol-of-vol, range amplitude, combo mom*effratio, sharpe."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from miner1_common import load_close, ic_analysis, coverage, turnover

t0 = time.time()
closes = load_close()
idx = None
for s in closes:
    idx = closes[s].index if idx is None else idx.intersection(closes[s].index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]
print(f"loaded {len(idx)} common dates {idx.min().date()}..{idx.max().date()} ({time.time()-t0:.1f}s)")

CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in closes})
OP = pd.DataFrame({s: closes[s]["open"].reindex(idx).astype(float) for s in closes})
HI = pd.DataFrame({s: closes[s]["high"].reindex(idx).astype(float) for s in closes})
LO = pd.DataFrame({s: closes[s]["low"].reindex(idx).astype(float) for s in closes})
VOL = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in closes})
RET = CP.pct_change()
LRET = np.log(CP / CP.shift(1))
SYMS = list(closes.keys())


def run(name, panel, verbose=True):
    panel = panel.reindex(idx)
    c = float(panel.notna().sum().sum()) / (len(idx) * len(SYMS))
    to = turnover(panel)
    ic1 = ic_analysis(panel, closes, fwd_days=1)
    ic5 = ic_analysis(panel, closes, fwd_days=5)
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    if verbose:
        print(f"{name:26s} cov={c:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
              f"| {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": c, "to": to, "ic1": ic1, "ic5": ic5, "passed": passed}


panels = {}

# --- rolling skewness / kurtosis of daily log returns ---
for n in (20, 60, 120):
    panels[f"skew_{n}d"] = LRET.rolling(n).skew()
for n in (60, 120):
    panels[f"kurt_{n}d"] = LRET.rolling(n).kurt()

# --- autocorrelation of returns (trend persistence) ---
for n in (20, 60):
    m = LRET.rolling(n).mean()
    v = LRET.rolling(n).std()
    ac = (LRET - m) * (LRET.shift(1) - m.shift(1))
    ac = ac.rolling(n).mean() / (v * v.shift(1))
    panels[f"autocorr_{n}d"] = ac

# --- Kaufman efficiency ratio: |close-close_n| / sum|rets| ---
for n in (20, 60, 120):
    num = (CP - CP.shift(n)).abs()
    den = LRET.abs().rolling(n).sum()
    panels[f"effratio_{n}d"] = num / den

# --- trend slope t-stat from linear regression on log price ---
def slope_tstat(n):
    out = {}
    for s in SYMS:
        lp = np.log(CP[s])
        x = np.arange(n)
        xm = x - x.mean()
        xx = (xm ** 2).sum()
        b = lp.rolling(n).apply(lambda w: np.polyfit(x, w, 1)[0] if np.isfinite(w).all() else np.nan, raw=True)
        resid = lp.rolling(n).apply(lambda w: np.polyval(np.polyfit(x, w, 1), x) - w if np.isfinite(w).all() else np.nan, raw=True)
        se = resid.rolling(n).std() / np.sqrt(xx)
        out[s] = b / se
    return pd.DataFrame(out)

panels["slope_t_60d"] = slope_tstat(60)
panels["slope_t_120d"] = slope_tstat(120)

# --- R2 of trend fit ---
def trend_r2(n):
    out = {}
    for s in SYMS:
        lp = np.log(CP[s])
        x = np.arange(n)
        out[s] = lp.rolling(n).apply(
            lambda w: np.corrcoef(x, w)[0, 1] ** 2 if np.isfinite(w).all() else np.nan, raw=True)
    return pd.DataFrame(out)

panels["trend_r2_60d"] = trend_r2(60)

# --- downside/upside vol ratio ---
def du_ratio(n):
    down = (LRET.clip(upper=0) ** 2).rolling(n).mean().apply(np.sqrt)
    up = (LRET.clip(lower=0) ** 2).rolling(n).mean().apply(np.sqrt)
    return down / up

panels["du_ratio_60d"] = du_ratio(60)
panels["du_ratio_120d"] = du_ratio(120)

# --- overnight gap: open/prev_close - 1 ---
panels["gap_1d"] = OP / CP.shift(1) - 1.0
panels["gap_5d_avg"] = (OP / CP.shift(1) - 1.0).rolling(5).mean()

# --- volume-price correlation (rolling corr of LRET vs dlog vol) ---
dvol = np.log(VOL + 1e-9).diff()
for n in (20, 60):
    panels[f"vpcorr_{n}d"] = LRET.rolling(n).corr(dvol)

# --- vol-of-vol: rolling std of 20d realized vol ---
rv20 = LRET.rolling(20).std()
panels["volofvol_60d"] = rv20.rolling(60).std()

# --- range amplitude ---
panels["range_20d_avg"] = ((HI - LO) / CP).rolling(20).mean()
panels["range_60d_avg"] = ((HI - LO) / CP).rolling(60).mean()

# --- combo: 60d momentum * effratio (trend quality weighted momentum) ---
mom60 = LRET.rolling(60).sum()
panels["mom60_x_eff60"] = mom60 * panels["effratio_60d"]

# --- sharpe ratio (60d mean/std) ---
panels["sharpe_60d"] = LRET.rolling(60).mean() / LRET.rolling(60).std()

results = []
for name, p in panels.items():
    results.append(run(name, p))

print(f"\ntotal runtime {time.time()-t0:.1f}s")
