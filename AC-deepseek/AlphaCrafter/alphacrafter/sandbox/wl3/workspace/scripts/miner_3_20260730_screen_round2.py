"""miner_3 2026-07-30 screening round 2: beta-to-anchor and idiosyncratic-trend factors.

Winning family so far is beta-to-risk-anchor (spx_beta_60 IC .080, btc_beta_60
IC .034) and conditional FX beta (dxy_beta_cond). Screen new, distinct members:
  - xau_beta_60     : 60d beta of asset returns to XAU (safe-haven) returns
  - wti_beta_60     : 60d beta to WTI (commodity cycle)
  - us10y_beta_60   : 60d beta to US10Y returns (rate sensitivity)
  - hs300_beta_60   : 60d beta to 000300.SH (China exposure)
  - ndx_beta_60     : 60d beta to NDX (tech beta) - corr check vs spx_beta_60
  - spx_rel_mom_60  : 60d momentum (skip 5) minus SPX 60d momentum (idiosyncratic)

Correlation audit uses extended library: 4 recomputed library factors + 7
artifact-bearing effective factors (bollinger_z_20d, dxy_beta_cond_60x20,
high_low_range_pos_20, rsi_14d, btc_beta_60, eurusd_beta_cond_60x20,
spx_beta_60) so new factors are not redundant with the live library.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, factor_to_panel,
                           validate_factor, max_library_correlation,
                           canonical_grid, VAL_START, VAL_END,
                           WATCHLIST, build_library_panels)

prices = load_prices(days=2000)
print(f"loaded {len(prices)} assets")

def beta_anchor(anchor_df):
    def fn(df, s):
        if anchor_df is None or len(anchor_df) < 70:
            return None
        r = df['close'].pct_change()
        ar = anchor_df['close'].pct_change()
        z = pd.concat([r.rename('r'), ar.rename('a')], axis=1).dropna()
        b = z['r'].rolling(60).cov(z['a']) / z['a'].rolling(60).var()
        return b.reindex(z.index)
    return fn

def load_artifact_panels():
    """Load effective artifact-bearing factor matrices back into panels."""
    out = {}
    grid = canonical_grid(prices)
    for fid in ['bollinger_z_20d', 'dxy_beta_cond_60x20', 'high_low_range_pos_20',
                'rsi_14d', 'btc_beta_60', 'eurusd_beta_cond_60x20', 'spx_beta_60']:
        try:
            art = np.load(f'factors/{fid}_signal.npy', allow_pickle=False)
            out[fid] = pd.DataFrame(art, index=grid, columns=WATCHLIST)
        except Exception as e:
            print('  artifact load fail', fid, e)
    return out

anchors = {name: prices[sym] for name, sym in
           [('xau', 'XAU'), ('wti', 'WTI'), ('us10y', 'US10Y'),
            ('hs300', '000300.SH'), ('ndx', 'NDX'), ('spx', 'SPX')]}

library_panels = build_library_panels(prices)
library_panels.update(load_artifact_panels())
print('extended library size:', len(library_panels))

candidates = {
    'xau_beta_60': beta_anchor(anchors['xau']),
    'wti_beta_60': beta_anchor(anchors['wti']),
    'us10y_beta_60': beta_anchor(anchors['us10y']),
    'hs300_beta_60': beta_anchor(anchors['hs300']),
    'ndx_beta_60': beta_anchor(anchors['ndx']),
}

def spx_rel_mom_60(df, s):
    if anchors['spx'] is None:
        return None
    m = df['close'].shift(5) / df['close'].shift(65) - 1.0
    ms = anchors['spx']['close'].shift(5) / anchors['spx']['close'].shift(65) - 1.0
    return (m - ms).reindex(df.index)

candidates['spx_rel_mom_60'] = spx_rel_mom_60

results = {}
for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data'); continue
    rho, best = max_library_correlation(panel, library_panels)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = best
    results[fid] = (m, panel)
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    print(f"Factor {fid}: panel {panel.shape} range {panel.index.min().date()}..{panel.index.max().date()}")
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=1, default=str))
    print("decay:", json.dumps(m['decay_ic_by_horizon'], default=str))
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f} |ICIR|={abs(m['icir']):.4f} -> {'PASS' if ok else 'FAIL'}  max_corr={rho:.3f} vs {best}")
    print()
