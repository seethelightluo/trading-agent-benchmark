"""miner_1 2026-08-13: corrected validation of autocorr_20.

Root cause of prior degenerate coverage: the shared GRID is the UNION calendar
(weekends included because BTC/ETH trade 7d). Computing ret*ret.shift(1) on the
union grid NaNs every Monday for weekday assets -> 99% NaN panel -> only 8 IC dates.

Fix: compute the factor per-asset on its OWN calendar (consecutive sessions),
then reindex to GRID for cross-sectional IC on dates with >=8 valid instruments.

Factor: lag-1 return persistence proxy  mean(r_t*r_{t-1})/mean(r_t^2) over 20d.
Gates: |IC|>=0.0070 AND |ICIR|>=0.0840 at h=10; library corr reported for audit.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DS = json.load(open("../persistent/date.json"))
TRADING_DAYS = DS["trading_days"]
VISIBLE = DS["visible_through"]
ROW0 = TRADING_DAYS.index("2020-01-01")
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
print(f"grid rows: {len(GRID)}  {GRID[0]}..{GRID[-1]}  visible={VISIBLE}")

TRADABLES = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
             "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DATA_DIR = Path("../persistent/stock_data")
HORIZON = 10
MIN_ASSETS = 8
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def load_close_own(sym):
    df = pd.read_csv(DATA_DIR / f"{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)].sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    return pd.to_numeric(df["close"], errors="coerce").astype(float)


CLOSE_OWN = {a: load_close_own(a) for a in TRADABLES}
for a in TRADABLES:
    print(f"{a:10s} own-calendar rows={len(CLOSE_OWN[a]):5d}  "
          f"nan={CLOSE_OWN[a].isna().mean():.3f}  {CLOSE_OWN[a].index[0]}..{CLOSE_OWN[a].index[-1]}")


def per_asset_factor(fn):
    """fn(close_series) -> factor series on own calendar; reindex to GRID."""
    out = {}
    for a in TRADABLES:
        s = CLOSE_OWN[a]
        r = s.pct_change()
        num = (r * r.shift(1)).rolling(20, min_periods=10).mean()
        den = (r ** 2).rolling(20, min_periods=10).mean()
        f = num / np.where(np.abs(den) < 1e-12, np.nan, den)
        out[a] = f.reindex(GRID)
    return pd.DataFrame(out, index=GRID)


F = per_asset_factor(None)
per_day = F.notna().sum(axis=1)
print("\nper-date valid assets: min=%d median=%d max=%d, dates>=8: %d / %d" %
      (per_day.min(), int(per_day.median()), per_day.max(), int((per_day >= 8).sum()), len(per_day)))
print("factor valid frac by asset:")
print(F.notna().mean().round(3).to_string())
cov_a = float(F.notna().sum().sum() / (F.shape[0] * F.shape[1]))
cov_d = float((per_day >= MIN_ASSETS).mean())
print(f"coverage_asset_days={cov_a:.4f} coverage_dates_ge8={cov_d:.4f}")

# ---- forward returns on own calendar, reindexed ----
def fwd_panel(h):
    out = {}
    for a in TRADABLES:
        s = CLOSE_OWN[a]
        out[a] = (s.shift(-h) / s - 1.0).reindex(GRID)
    return pd.DataFrame(out, index=GRID)


def ic_series(fp, rp):
    dates = fp.index.intersection(rp.index)
    Fr = fp.loc[dates].rank(axis=1).values
    Rr = rp.loc[dates].rank(axis=1).values
    m = (~np.isnan(Fr)) & (~np.isnan(Rr))
    valid = m.sum(axis=1) >= MIN_ASSETS
    ics = np.full(len(dates), np.nan)
    for i in np.where(valid)[0]:
        f = Fr[i, m[i]] - Fr[i, m[i]].mean()
        r = Rr[i, m[i]] - Rr[i, m[i]].mean()
        d = np.sqrt((f * f).sum() * (r * r).sum())
        ics[i] = (f * r).sum() / d if d > 0 else np.nan
    return pd.Series(ics, index=dates)


FWD = {h: fwd_panel(h) for h in (1, 2, 3, 5, 10, 20)}
ic10 = ic_series(F, FWD[10]).dropna()
icir = ic10.mean() / ic10.std() if len(ic10) > 1 and ic10.std() > 0 else np.nan
print(f"\nh=10 IC series: n_dates={len(ic10)}  mean_ic={ic10.mean():+.4f}  std={ic10.std():.4f}  "
      f"ICIR={icir:+.4f}")
print("IC first/last:", ic10.index[0], ic10.index[-1])

ranked = F.rank(axis=1, pct=True)
tvals = []
for i in range(10, len(ranked), 10):
    a, b = ranked.iloc[i - 10], ranked.iloc[i]
    m = a.notna() & b.notna()
    if m.sum() >= MIN_ASSETS:
        tvals.append(float((b[m] - a[m]).abs().mean()))
to = float(np.mean(tvals)) if tvals else np.nan
print(f"turnover_10d_rank={to:.4f} (n={len(tvals)})")

decay = {str(h): round(float(ic_series(F, FWD[h]).dropna().mean()), 4) for h in (1, 2, 3, 5, 10, 20)}
print("decay:", decay)
print(f"ic_last250={ic10.iloc[-250:].mean():+.4f} ic_last60={ic10.iloc[-60:].mean():+.4f}")

# ---- library correlation ----
LIB = {}
for f in sorted(Path("factors").glob("*.signal.npy")):
    try:
        a = np.load(f, allow_pickle=True)
        if a.shape[1] == 15:
            n = min(a.shape[0], len(GRID))
            LIB[f.stem.replace(".signal", "")] = pd.DataFrame(a[:n], index=GRID[:n], columns=TRADABLES)
    except Exception as e:
        print("skip", f, e)
print("library artifacts:", len(LIB))


def panel_rank_corr(a, b):
    dates = a.index.intersection(b.index)
    Ar = a.loc[dates].rank(axis=1).values
    Br = b.loc[dates].rank(axis=1).values
    m = (~np.isnan(Ar)) & (~np.isnan(Br))
    valid = m.sum(axis=1) >= MIN_ASSETS
    cs = []
    for i in np.where(valid)[0]:
        x = Ar[i, m[i]] - Ar[i, m[i]].mean()
        y = Br[i, m[i]] - Br[i, m[i]].mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        if d > 0:
            cs.append((x * y).sum() / d)
    return float(np.mean(cs)) if cs else 0.0


libc = {fid: panel_rank_corr(F, sig) for fid, sig in LIB.items()}
top = sorted(libc.items(), key=lambda kv: -abs(kv[1]))[:8]
print("\ntop library correlations:")
for k, v in top:
    print(f"  {k:28s} rho={v:+.4f}")
maxabs = max((abs(v) for v in libc.values()), default=0.0)
print(f"max_abs_library_correlation = {maxabs:.4f}")

passed = abs(ic10.mean()) >= GATE_IC and abs(icir) >= GATE_ICIR
print(f"\nGATE: IC={ic10.mean():+.4f} (>={GATE_IC}) ICIR={icir:+.4f} (>={GATE_ICIR}) "
      f"n_dates={len(ic10)} -> {'PASS' if passed else 'FAIL'}")

if passed:
    sig = F.copy()
    np.save("factors/autocorr_20.signal.npy", sig.values.astype(float))
    print("saved factors/autocorr_20.signal.npy", sig.values.shape,
          "nan% {:.3f}".format(float(np.isnan(sig.values).mean())))
else:
    print("GATE FAIL -> did NOT save signal artifact")

out = {
    "visible": VISIBLE, "grid_rows": len(GRID), "n_ic_dates": int(len(ic10)),
    "ic": round(float(ic10.mean()), 4), "icir": round(float(icir), 4),
    "ic_hit_ratio": round(float((np.sign(ic10) == np.sign(ic10.mean())).mean()), 3),
    "coverage_asset_days": round(cov_a, 4), "coverage_dates_ge8": round(cov_d, 4),
    "turnover_10d_rank": round(to, 4), "decay_ic_by_horizon": decay,
    "ic_last250": round(float(ic10.iloc[-250:].mean()), 4),
    "ic_last60": round(float(ic10.iloc[-60:].mean()), 4),
    "max_abs_library_correlation": round(maxabs, 4),
    "library_pairwise_corr_top": {k: round(v, 4) for k, v in top},
    "pass": bool(passed),
}
json.dump(out, open("scripts/miner_1_20260813_autocorr20_validation_fixed.json", "w"), indent=1)
print("saved scripts/miner_1_20260813_autocorr20_validation_fixed.json")
