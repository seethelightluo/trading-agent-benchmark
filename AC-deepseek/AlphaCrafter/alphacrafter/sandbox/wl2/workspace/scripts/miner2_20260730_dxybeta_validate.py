"""miner_2: validate dxy_beta_cond_60x20 candidate factor.
Idea: conditional DXY sensitivity. For each asset, estimate rolling 60d beta to
DXY returns, then multiply by the trailing 20d DXY move. When DXY appreciates,
assets positively exposed to USD strength should outperform (and vice versa).
Admission gates: |IC| >= 0.0070, |ICIR| >= 0.0840 (shared benchmark-wide).
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner2_20260730_factorlib import (load_panel, load_visible_through,
                                       TRADABLE, OBS, full_validate)

END = load_visible_through()
print("visible_through:", END.date())

panel = load_panel(TRADABLE, "stock", END)
ret = panel.pct_change()
macro = load_panel(OBS, "index", END)
dxy = macro["DXY"]

# --- candidate: dxy_beta_cond_60x20 ---
def rolling_beta(ret_df, macro_ret, win=60):
    out = pd.DataFrame(index=ret_df.index, columns=ret_df.columns, dtype=float)
    for sym in ret_df.columns:
        z = pd.concat([ret_df[sym].rename("a"), macro_ret.rename("m")], axis=1).dropna()
        out[sym] = z["a"].rolling(win).cov(z["m"]) / z["m"].rolling(win).var()
    return out

beta = rolling_beta(ret, dxy.pct_change(), 60)
dxy_mom = dxy / dxy.shift(20) - 1.0
factor = beta * dxy_mom
factor = factor.replace([np.inf, -np.inf], np.nan)

print("factor shape:", factor.shape, "| non-null:", int(factor.notna().sum().sum()))
m = full_validate(factor, panel, horizon=10, direction=1, label="dxy_beta_cond_60x20")
if m is not None:
    print(json.dumps(m, default=str, indent=1))
    ok = abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840
    print("ADMISSION GATE:", "PASS" if ok else "FAIL",
          f"(|IC|={abs(m['ic']):.4f}>=0.0070, |ICIR|={abs(m['icir']):.4f}>=0.0840)")

# decay at other horizons
ret_panel = panel.pct_change()
for h in (1, 3, 5, 10, 20):
    from miner2_20260730_factorlib import factor_metrics
    fwd = ret_panel.shift(-h)
    mm = factor_metrics(factor, fwd, h, min_assets=8, direction=1)
    if mm:
        print(f"  h={h:>2} IC={mm['ic']:.4f} ICIR={mm['icir']:.4f} ndates={mm['n_ic_dates']}")
