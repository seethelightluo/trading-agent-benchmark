"""
Factor: momentum_reversal_ratio
Idea: Ratio of short-term momentum (5d) to medium-term momentum (20d).
- Ratio > 1: recent momentum accelerating -> continuation
- Ratio 0<ratio<1: decelerating momentum
- Negative: direction reversal
Expected direction: 1 (higher ratio = positive forward returns)
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

acct = get_account_dict()
watchlist = acct.get('watch_list', [])
print(f"Watchlist size: {len(watchlist)}")

data = {}
for sym in watchlist:
    df = get_stock_daily_data(symbol=sym, days=600)
    if df is not None:
        data[sym] = df

prices = pd.DataFrame({sym: data[sym]['close'].values for sym in data},
                      index=data[watchlist[0]]['date'].values)
returns = prices.pct_change()
print(f"Price data shape: {prices.shape}, dates {prices.index[0]} to {prices.index[-1]}")

SHORT, MEDIUM, HORIZON = 5, 20, 5
r5 = (1 + returns).rolling(SHORT).apply(np.prod, raw=True) - 1
r20 = (1 + returns).rolling(MEDIUM).apply(np.prod, raw=True) - 1
ratio = r5 / r20.abs()
ratio = ratio.replace([np.inf, -np.inf], np.nan)

fwd = (1 + returns).rolling(HORIZON).apply(np.prod, raw=True).shift(-HORIZON) - 1
# alternatively use shift approach for clean forward returns
# fwd = (1 + returns.shift(-HORIZON-1)).rolling(HORIZON).apply(np.prod, raw=True) - 1
# Use simpler: fwd return over next HORIZON days from t to t+HORIZON
fwd = prices.shift(-HORIZON) / prices - 1

factor_values = ratio
fwd_values = fwd

MIN_VALID = 8
ic_list, n_assets_list = [], []
common_idx = factor_values.notna().any(axis=1) & fwd_values.notna().any(axis=1)
common_idx = common_idx[common_idx].index

for idx in common_idx:
    f_row = factor_values.loc[idx]
    r_row = fwd_values.loc[idx]
    valid = f_row.notna() & r_row.notna()
    n_valid = valid.sum()
    if n_valid >= MIN_VALID:
        f_vals = f_row[valid].values
        r_vals = r_row[valid].values
        if len(np.unique(f_vals)) > 1 and len(np.unique(r_vals)) > 1:
            ic, _ = spearmanr(f_vals, r_vals)
            if not np.isnan(ic):
                ic_list.append(ic)
                n_dates_list.append(n_valid)

ic_array = np.array(ic_list)
mean_ic = ic_array.mean()
std_ic = ic_array.std()
icir = mean_ic / std_ic if std_ic > 0 else np.nan
hit = (ic_array > 0).mean()
print(f"\n=== VALIDATION ===")
print(f"Valid IC dates: {len(ic_array)}")
print(f"Mean IC: {mean_ic:.5f}, ICIR: {icir:.5f}, IC hit ratio: {hit:.4f}")
print(f"Avg assets/date: {np.mean(n_dates_list):.1f}")

coverage = ratio.notna().mean().mean()
print(f"Coverage: {coverage:.4f}")

# Turnover from rank correlation
rc_list = []
for i in range(1, len(common_idx)):
    p, c = common_idx[i-1], common_idx[i]
    pr, cr = factor_values.loc[p], factor_values.loc[c]
    valid = pr.notna() & cr.notna()
    if valid.sum() >= MIN_VALID:
        pv, cv = pr[valid].values, cr[valid].values
        if len(np.unique(pv)) > 1 and len(np.unique(cv)) > 1:
            rc, _ = spearmanr(pv, cv)
            if not np.isnan(rc):
                rc_list.append(rc)
print(f"Mean rank corr: {np.mean(rc_list):.4f}, Turnover(1-rc): {1-np.mean(rc_list):.4f}")

# Decay analysis
print("\n=== DECAY ===")
for h in [1, 2, 3, 5, 10, 20]:
    fwd_h = prices.shift(-h) / prices - 1
    ics = []
    for idx in common_idx:
        f_row = factor_values.loc[idx]
        r_row = fwd_h.loc[idx]
        valid = f_row.notna() & r_row.notna()
        if valid.sum() >= MIN_VALID:
            f_vals, r_vals = f_row[valid].values, r_row[valid].values
            if len(np.unique(f_vals)) > 1 and len(np.unique(r_vals)) > 1:
                icv, _ = spearmanr(f_vals, r_vals)
                if not np.isnan(icv):
                    ics_h.append(icv)
    if ics_h:
        print(f"horizon {h}: IC={np.mean(ics_h):.5f} icir={np.mean(ics_h)/np.std(ics_h):.4f}")

IC_THRESH, ICIR_THRESH = 0.0070, 0.0840
passed = abs(mean_ic) >= IC_THRESH and abs(icir) >= ICIR_THRESH
print(f"\nGate: |IC|={abs(mean_ic):.5f} vs {IC_THRESH}, |ICIR|={abs(icir):.5f} vs {ICIR_THRESH}")
print(f"OVERALL: {'EFFECTIVE' if passed else 'REJECTED'}")