"""miner_1 exploration round 2: fix rs/beta/tom bugs, add pos_freq family + composites."""
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
VOL = pd.DataFrame({s: closes[s]["volume"].reindex(idx).astype(float) for s in closes})
RET = CP.pct_change()
LRET = np.log(CP / CP.shift(1))
EW = CP.mean(axis=1)
EWR = EW.pct_change()
EWRdf = pd.DataFrame({s: EWR for s in closes})
SYMS = list(closes.keys())
fwd = {h: CP.shift(-h) / CP - 1.0 for h in (1, 2, 3, 5, 10)}


def cov_to(factor_df):
    # coverage over the validation window only
    tot = factor_df.notna().sum().sum()
    return tot / (len(idx) * len(SYMS)), turnover(factor_df)


def run(name, panel, verbose=True):
    panel = panel.reindex(idx)
    c, t = cov_to(panel)
    ic1 = ic_analysis(panel, closes, fwd_days=1)
    ic5 = ic_analysis(panel, closes, fwd_days=5)
    ic10 = ic_analysis(panel, closes, fwd_days=10)
    passed = (abs(ic1["ic"]) >= 0.0070) and (abs(ic1["icir"]) >= 0.0840)
    if verbose:
        print(f"{name:26s} cov={c:.3f} to={t:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
              f"hit1={ic1['hit']:.2f} n1={ic1['n_dates']} | IC5={ic5['ic']:+.4f} ICIR5={ic5['icir']:+.3f} "
              f"| IC10={ic10['ic']:+.4f} | {'PASS' if passed else 'fail'}")
    return {"name": name, "cov": c, "to": t, "ic1": ic1, "ic5": ic5, "ic10": ic10, "passed": passed}


panels = {}

# ---------- pos_freq family (fraction of up days) ----------
for n in (120, 180, 250, 300, 400):
    panels[f"pos_freq_{n}d"] = (LRET > 0).rolling(n).mean()
for n in (180, 250):
    panels[f"pos_freq_{n}d_skip5"] = (LRET > 0).rolling(n).mean().shift(5)
panels["pos_freq_250d_skip20"] = (LRET > 0).rolling(250).mean().shift(20)

# ---------- volume z (attention) ----------
lvol = np.log(VOL + 1e-9)
for n in (60, 90, 120):
    panels[f"vol_z_{n}d"] = (lvol - lvol.rolling(n).mean()) / lvol.rolling(n).std()

# ---------- fixed rs / beta / macro ----------
for n in (20, 60, 120):
    panels[f"rs_{n}d"] = LRET.rolling(n).sum() - EWRdf.rolling(n).sum()

vix = pd.read_csv("../persistent/index_data/VIX.csv")
vix["date"] = pd.to_datetime(vix["date"])
vix = vix.set_index("date")["close"].reindex(idx)
spx = CP["SPX"]
spx_trend = spx / spx.shift(20) - 1.0
vix_trend = vix / vix.shift(20) - 1.0


def roll_beta(y, x, win):
    m = pd.concat([y, x], axis=1).dropna()
    yv, xv = m.iloc[:, 0], m.iloc[:, 1]
    return (yv.rolling(win).cov(xv) / xv.rolling(win).var())


b_spx = {w: pd.DataFrame({s: roll_beta(RET[s], spx.pct_change(), w) for s in SYMS}) for w in (60, 120)}
b_vix = {w: pd.DataFrame({s: roll_beta(RET[s], vix.pct_change(), w) for s in SYMS}) for w in (60, 120)}

panels["beta_spx60_x_spxtrend"] = b_spx[60] * np.sign(spx_trend)
panels["beta_vix60_x_vixret"] = -b_vix[60] * np.sign(vix_trend)
panels["beta_spx60_x_vixlevel"] = b_spx[60] * np.sign(vix.rolling(20).mean() - vix.rolling(120).mean())
panels["beta_vix120_x_vixtrend"] = -b_vix[120] * np.sign(vix_trend)

# ---------- composites (rank-average) ----------
def zrank(panel):
    return panel.rank(axis=1)


mom = {n: LRET.rolling(n).sum() for n in (20, 60, 120, 250)}
norm_ma20_60 = (CP.rolling(20).mean() - CP.rolling(60).mean()) / CP.rolling(60).std()
close_vs_ma120 = CP / CP.rolling(120).mean() - 1.0
dist_52w_low = CP / CP.rolling(252).min() - 1.0

pf = (LRET > 0).rolling(250).mean()
panels["comp_posfreq_mom"] = (zrank(pf) + zrank(mom[250]) + zrank(dist_52w_low)) / 3
panels["comp_posfreq_mom60"] = (zrank(pf) + zrank(mom[60].shift(5)) + zrank(norm_ma20_60)) / 3
panels["comp_mom_rank"] = (zrank(mom[20]) + zrank(mom[60]) + zrank(mom[120]) + zrank(mom[250])) / 4
panels["comp_trend_mix"] = (zrank(mom[60].shift(5)) + zrank(norm_ma20_60) + zrank(close_vs_ma120) + zrank(dist_52w_low)) / 4
panels["comp_mom_er"] = (zrank(mom[60]) + zrank((CP - CP.shift(60)).abs() / LRET.abs().rolling(60).sum()) + zrank(pf)) / 3
panels["comp_mom_rs"] = (zrank(mom[60]) + zrank(panels["rs_60d"]) + zrank(pf)) / 3

# ---------- regime-conditional ----------
ew_up = (EWR.rolling(20).sum() > 0).astype(float)
panels["switch_mom_rev"] = mom[60].shift(5) * ew_up + (-LRET.rolling(5).sum()) * (1 - ew_up)
panels["pf250_ewup"] = pf * ew_up
panels["pf250_x_ewtrend"] = pf * np.sign(EWR.rolling(20).sum())

res = [run(n, p) for n, p in panels.items()]
print(f"\nfinished {time.time()-t0:.1f}s | {sum(r['passed'] for r in res)} passed gate")
for r in res:
    if r["passed"]:
        print("PASS:", r["name"], "IC1=", r["ic1"]["ic"], "ICIR1=", r["ic1"]["icir"])
