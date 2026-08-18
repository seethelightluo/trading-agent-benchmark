"""Trader 2029-09-25: sync strategy.py docstring to current 7-factor ensemble (logic unchanged, dynamic load)."""
import pathlib

p = pathlib.Path("strategy.py")
t = p.read_text()

# replace the module docstring (first triple-quoted block)
start = t.index('"""')
end = t.index('"""', start + 3) + 3

new_doc = '''"""Trader strategy - 7-factor cross-asset ensemble (2029-09-25 cycle).

Ensemble (factor_ensemble.json, quality_ic_tilt, loaded dynamically):
  rel_mom_20d_skip5       0.24  (dir+1, BOOSTED top - confirmed recovery
                                 momentum SPX/BTC/WTI 20d positive; capped
                                 below 0.30 while VIX 72.6 not <30)
  downside_vol_ratio_20   0.21  (dir+1, TRIM from 0.28 - bear-tape champion
                                 stays core but trimmed as recovery firms)
  beta_ew_60d             0.17  (dir+1, BOOST - recovery beta capture,
                                 low turnover 1.25 suits high-vol tape)
  max_ret_20d             0.12  (dir+1, upside-lottery on strong movers)
  corr_ew_60              0.10  (dir+1, high-correlation regime keeps relevant)
  kurt_20d_skip5          0.08  (dir+1, tail-risk diversifier, kept small)
  dxy_beta_cond_60x20     0.08  (dir -1: DXY firmed +1.1%/20d -> hedge USD
                                 strength; reverted from Jul +1 flip)

Weights are loaded dynamically from factor_ensemble.json at runtime.
Long-only, fully invested across the 15 watchlist assets (no cash sleeve).

Regime context (data thru 2029-09-24): sideways-to-recovering within a
deep-bear tape. VIX 72.6 EXTREME (down from 88.4/82.3 peak, still >30 ->
risk-off trigger HIT -> def_floor 0.12 auto high end); SPX +7.9%/60d,
BTC +27.7%/60d, WTI +30.9%/60d beta bounce; HSI -12.2%/60d, XAU -7.1%/60d
weak. US10Y 4.05% (-2.1%/20d), DXY 104.7 (+1.1%/20d firming). FROZEN feeds:
NDX, SOX, 000688.SH, CN10Y (~33% dead weight, structural drag).

Defensive floor XAU-anchored (2028-12-05 updated):
- XAU 70% of defensive budget (primary floor); US10Y 15% cap (bond floor
  bled persistently through 2028 yield up-cycle, cap CUT 25%->15%);
  CN10Y frozen -> 0. Global per-asset cap 0.15 trims XAU to 15% max.

Per-asset cap 0.15; submits one complete target via rebalance_to_weights helper.
"""'''

t = t[:start] + new_doc + t[end:]
p.write_text(t)
print("docstring synced; file length:", len(t))
