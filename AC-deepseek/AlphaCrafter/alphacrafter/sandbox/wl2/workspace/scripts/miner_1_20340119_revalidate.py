"""miner_1 2034-01-19 factor library re-validation harness.

Loads persisted factor definitions (calculation.expression) from factors/*.json,
evaluates them against close panels through 2034-01-18 (previous completed day),
computes IC / ICIR / hit / coverage at admission horizon 10, plus a trailing
365d window to detect drift. No future data: forward returns use shift(-h).
"""
import json, glob, re, sys
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
            print('WARN no data', s)
            continue
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
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

def summarize(ic_s):
    if len(ic_s) < 5:
        return None
    ic = float(ic_s.mean()); icir = float(ic_s.mean()/ic_s.std()) if ic_s.std() > 0 else np.nan
    hit = float((ic_s > 0).mean())
    return {'ic': round(ic, 4), 'icir': round(icir, 3), 'hit': round(hit, 3), 'n': len(ic_s),
            'pass': bool(np.isfinite(ic) and np.isfinite(icir) and abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR)}

def safe_eval_expr(expr, C):
    """Evaluate a pandas expression string against close DataFrame C."""
    env = {'C': C, 'np': np, 'pd': pd}
    # translate common expressions: close -> C
    e = expr.replace('close', 'C')
    try:
        return eval(e, {'__builtins__': {}}, env)
    except Exception as ex:
        return None

def main():
    C = load_close()
    print('grid', C.shape, C.index.min().date(), '->', C.index.max().date())
    rets = C.pct_change()
    # build forward return panel at horizon 10
    fwd = C.shift(-HORIZON) / C - 1.0

    files = sorted(glob.glob('factors/*.json'))
    results = []
    for f in files:
        if 'ensemble' in f or 'bak' in f:
            continue
        try:
            d = json.load(open(f))
        except Exception as ex:
            print('SKIP unreadable', f, ex); continue
        fid = d.get('factor_id')
        expr = d.get('calculation', {}).get('expression')
        if not expr:
            print('SKIP no expr', f); continue
        panel = safe_eval_expr(expr, C)
        if panel is None or not hasattr(panel, 'shape') or panel.shape[1] != 15:
            # try to salvage: some expr reference .pct_change() etc -> wrap
            try:
                e2 = expr.replace('close', 'C')
                panel = eval(e2, {'__builtins__': {}}, {'C': C, 'np': np, 'pd': pd})
            except Exception:
                pass
        if panel is None or not hasattr(panel, 'shape'):
            print('FAIL eval', fid, expr[:60]); continue
        if isinstance(panel, pd.Series):
            panel = panel.to_frame(C.columns[0])
        panel = panel.reindex(columns=C.columns)
        fdf = panel.astype(float)
        ic_s = ic_series(fdf, fwd)
        full = summarize(ic_s)
        # trailing 365d
        tr = ic_s[ic_s.index >= ic_s.index[-1] - pd.Timedelta(days=365)] if len(ic_s) else ic_s
        recent = summarize(tr)
        cov = float(fdf.notna().mean().mean())
        results.append((fid, full, recent, cov, d.get('validation', {}).get('last_validated')))
        if full:
            print(f"{fid:28s} full ic={full['ic']:+.4f} icir={full['icir']:+.3f} hit={full['hit']:.3f} n={full['n']:4d} PASS={full['pass']} | 365d ic={recent['ic'] if recent else None:} icir={recent['icir'] if recent else None:} | cov={cov:.3f} last_validated={d.get('validation',{}).get('last_validated')}")

    print('\n=== GATE SUMMARY (full-sample admission horizon 10) ===')
    for fid, full, recent, cov, lv in results:
        if full:
            print(f"{fid:28s} PASS={full['pass']} ic={full['ic']:+.4f} icir={full['icir']:+.3f} | 365d ic={recent['ic'] if recent else float('nan'):+.4f} icir={recent['icir'] if recent else float('nan'):+.3f}")

if __name__ == '__main__':
    main()
