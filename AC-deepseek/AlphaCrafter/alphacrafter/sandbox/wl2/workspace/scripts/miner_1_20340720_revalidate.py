"""miner_1 2034-07-20 factor library re-validation (data through 2034-07-19).

Evaluates persisted factor definitions (calculation.expression) from factors/*.json
against close panels visible through 2034-07-19 (previous completed trading day).
IC / ICIR at admission horizon 10, full sample + trailing 365d + trailing 120d,
per-year regime splits. No future data (forward returns use shift(-h)).
"""
import json, glob
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
HORIZON = 10
MIN_ASSETS = 8
GATE_IC = 0.0070
GATE_ICIR = 0.0840

def load_close():
    out = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=4000)
        if df is None or len(df) < 300:
            print('WARN no data', s); continue
        df = df.copy(); df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        out[s] = df['close'].astype(float)
    idx = None
    for s, ser in out.items():
        idx = ser.index if idx is None else idx.union(ser.index)
    idx = idx.sort_values()
    for s in out:
        out[s] = out[s].reindex(idx)
    return pd.DataFrame(out)

def ic_series(factor_df, fwd_df):
    ics, dates = [], []
    for dt in factor_df.index:
        x = factor_df.loc[dt]; y = fwd_df.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_ASSETS:
            v = x[m].rank().corr(y[m].rank())
            if np.isfinite(v):
                ics.append(v); dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def summ(ic_s):
    if len(ic_s) < 5:
        return None
    ic = float(ic_s.mean()); icir = float(ic_s.mean()/ic_s.std()) if ic_s.std() > 0 else np.nan
    hit = float((ic_s > 0).mean())
    return {'ic': round(ic, 4), 'icir': round(icir, 3), 'hit': round(hit, 3), 'n': len(ic_s),
            'pass': bool(np.isfinite(ic) and np.isfinite(icir) and abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR)}

def eval_panel(expr, C):
    env = {'C': C, 'np': np, 'pd': pd}
    e = expr.replace('close', 'C')
    try:
        return eval(e, {'__builtins__': {}}, env)
    except Exception:
        return None

def main():
    C = load_close()
    print('grid', C.shape, C.index.min().date(), '->', C.index.max().date())
    fwd = C.shift(-HORIZON) / C - 1.0
    files = sorted(glob.glob('factors/*.json'))
    rows = []
    for f in files:
        if 'ensemble' in f or 'bak' in f:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        fid = d.get('factor_id')
        expr = d.get('calculation', {}).get('expression')
        if not expr:
            continue
        panel = eval_panel(expr, C)
        if panel is None or not hasattr(panel, 'shape'):
            print('FAIL eval', fid, expr[:60]); continue
        if isinstance(panel, pd.Series):
            panel = panel.to_frame(C.columns[0])
        panel = panel.reindex(columns=C.columns).astype(float)
        ic_s = ic_series(panel, fwd)
        full = summ(ic_s)
        last365 = summ(ic_s[ic_s.index >= ic_s.index[-1] - pd.Timedelta(days=365)]) if len(ic_s) else None
        last120 = summ(ic_s[ic_s.index >= ic_s.index[-1] - pd.Timedelta(days=120)]) if len(ic_s) else None
        cov = float(panel.notna().mean().mean())
        years = {}
        for y in range(2020, 2035):
            m = ic_s.index.year == y
            if m.sum() > 5:
                sub = ic_s[m]
                years[y] = f"{sub.mean():+.3f}/{sub.mean()/sub.std():+.2f}({m.sum()})"
        lv = d.get('validation', {}).get('last_validated')
        rows.append((fid, full, last365, last120, cov, lv, years))
        if full:
            print(f"{fid:28s} FULL ic={full['ic']:+.4f} icir={full['icir']:+.3f} hit={full['hit']:.3f} n={full['n']:4d} PASS={full['pass']} | "
                  f"365d ic={last365['ic'] if last365 else float('nan'):+.4f} icir={last365['icir'] if last365 else float('nan'):+.3f} | "
                  f"120d ic={last120['ic'] if last120 else float('nan'):+.4f} | cov={cov:.3f} lv={lv}")
    print('\n=== PER-YEAR IC (full sample) ===')
    for fid, full, _, _, cov, lv, years in rows:
        if full:
            print(f"{fid:28s} " + " ".join(f"{y}:{v}" for y, v in years.items()))
    print('\n=== GATE SUMMARY ===')
    for fid, full, last365, last120, cov, lv, _ in rows:
        if full:
            print(f"{fid:28s} FULL_PASS={full['pass']} ic={full['ic']:+.4f} icir={full['icir']:+.3f} | "
                  f"365d ic={last365['ic'] if last365 else float('nan'):+.4f} | 120d ic={last120['ic'] if last120 else float('nan'):+.4f}")

if __name__ == '__main__':
    main()
