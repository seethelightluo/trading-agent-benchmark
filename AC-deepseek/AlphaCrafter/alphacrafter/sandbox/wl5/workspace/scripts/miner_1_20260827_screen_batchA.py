"""miner_1 cycle 2026-08-27: screen batch A of new factor candidates.

Visible window restricted to VIS=2026-08-26 (last completed trading day before
current date 2026-08-27). 15-asset tradable cross-asset universe. All candidates
use close (or open/close) so coverage is full for all 15 assets.

Candidates (motivation):
  eff_ratio_30      Kaufman efficiency ratio: net move / total path length over 30d
                    (trend smoothness; complements signed R2 which uses linear fit).
  ret_autocorr_20   AR(1) autocorrelation of daily returns over 20d (persistence
                    of daily shocks vs mean reversion).
  vol_ts_5x60       vol(5d)/vol(60d)-1: short vs long volatility term structure.
  vol_ts_10x60      vol(10d)/vol(60d)-1.
  mom_accel_20x60   recent 20d return minus prior 20d return (60d-40d window):
                    trend acceleration/deceleration.
  intraday_drift_20 mean log(close/open) over 20d: intraday session drift
                    (complement of overnight gap).
  overnight_gap_20  mean open/prev_close-1 over 20d (re-validation with new VIS).
  var_ratio_5x60    variance ratio VR(5)=var(5d ret)/(5*var(1d ret)) over 60d:
                    mean-reversion (<1) vs momentum (>1) scaling behaviour.

Gate: |IC|>=0.007 and |ICIR|>=0.084 at h=10 on >=8 valid instruments.
"""
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import load_panel, closes_panel, forward_returns, ic_series, summary_metrics, regime_split

VIS = '2026-08-26'
H = 10
close = closes_panel(VIS)
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS}", flush=True)

src = load_panel(visible_through=VIS, source='stock')
opens = pd.DataFrame({s: df.set_index('date')['open'].astype(float) for s, df in src.items()}).sort_index()
opens = opens.reindex(columns=close.columns)
ret = close.pct_change()
lret = np.log(close).diff()

candidates = {}

# 1. Kaufman efficiency ratio 30d
path = ret.abs().rolling(30, min_periods=18).sum()
net = (close / close.shift(30) - 1.0).abs()
candidates['eff_ratio_30'] = net / path.replace(0, np.nan)

# 2. 1-lag autocorrelation of daily returns over 20d
r1 = ret.shift(1)
cov = ret.rolling(20, min_periods=8).cov(r1)
v0 = ret.rolling(20, min_periods=8).var()
v1 = r1.rolling(20, min_periods=8).var()
candidates['ret_autocorr_20'] = cov / np.sqrt(v0 * v1).replace(0, np.nan)

# 3/4. vol term structure
v5 = ret.rolling(5, min_periods=3).std()
v10 = ret.rolling(10, min_periods=6).std()
v60 = ret.rolling(60, min_periods=30).std()
candidates['vol_ts_5x60'] = v5 / v60 - 1.0
candidates['vol_ts_10x60'] = v10 / v60 - 1.0

# 5. momentum acceleration
mom20 = close / close.shift(20) - 1.0
mom60 = close / close.shift(60) - 1.0
candidates['mom_accel_20x60'] = mom20 - (mom60 - mom20)  # recent vs prior 20d (within 60d)

# 6. intraday drift
candidates['intraday_drift_20'] = np.log(close / opens).rolling(20, min_periods=8).mean()

# 7. overnight gap (re-validation)
gap = opens / close.shift(1) - 1.0
candidates['overnight_gap_20'] = gap.rolling(20, min_periods=8).mean()

# 8. variance ratio VR(5) over 60d
r5 = close.pct_change(5)
vr5 = r5.rolling(60, min_periods=30).var() / (5.0 * ret.rolling(60, min_periods=30).var()).replace(0, np.nan)
candidates['var_ratio_5x60'] = vr5 - 1.0

fr = forward_returns(close, H)
out = {}
for name, sig in candidates.items():
    sig = sig.reindex(close.index)
    ic_s = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic_s, sig, fr, close, h=H)
    if m is None:
        print(f"{name}: INSUFFICIENT dates (n_ic={len(ic_s)})", flush=True)
        continue
    m['regime'] = regime_split(ic_s)
    gate = bool(abs(m['ic']) >= 0.007 and abs(m['icir'] or 0) >= 0.084)
    out[name] = m
    print(f"=== {name}: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} "
          f"cov_ad={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} turn={m['turnover_10d_rank']} "
          f"GATE={gate}", flush=True)
    print(f"    decay: {m['decay_ic_by_horizon']}", flush=True)
    print(f"    regime: {m['regime']}", flush=True)

with open('scripts/miner_1_20260827_screen_batchA_results.json', 'w') as f:
    json.dump({'visible_through': VIS, 'horizon': H, 'candidates': out}, f, indent=1, default=str)
print("saved scripts/miner_1_20260827_screen_batchA_results.json", flush=True)
