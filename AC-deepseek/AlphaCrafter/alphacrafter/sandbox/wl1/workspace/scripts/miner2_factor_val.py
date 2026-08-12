"""miner2 shared validation helpers (2029-02-16). Daily cross-sectional Spearman IC vs forward returns."""
import pandas as pd, numpy as np

GATE_IC = 0.0070
GATE_ICIR = 0.0840
MIN_VALID = 8


def load_panel(path='scripts/miner2_panel_20290216.pkl'):
    return pd.read_pickle(path)


def library_signals(panel):
    """Recompute signal matrices of existing effective library factors (for pairwise rho audit)."""
    close = panel['close']
    high = panel['high']; low = panel['low']; open_ = panel['open']; vol = panel['vol']
    ret = panel['ret']
    lnc = np.log(close)
    sig = {}
    for nd in (1, 2, 3, 5):
        sig[f'rev_{nd}d'] = -(lnc - lnc.shift(nd))
        rng = high.rolling(nd).max() - low.rolling(nd).min()
        sig[f'nclv_{nd}d'] = -(close - low.rolling(nd).min()) / rng
    sig['nbody_1d'] = -(close - open_) / (high - low)
    sig['id_rev_1d'] = -(close / open_ - 1.0)
    rv20 = ret.rolling(20).std()
    sig['rev_1d_vs'] = -(lnc - lnc.shift(1)) / rv20
    sig['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1.0
    sig['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
    macro = panel['macro']
    vix = macro['VIX']
    vix_ret = vix.pct_change()
    beta_vix = ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
    sig['vix_beta_cond_60x20'] = -beta_vix * (vix / vix.shift(20) - 1.0)
    return sig


def daily_ic_series(signal, fwd_ret, min_valid=MIN_VALID):
    """Spearman rank IC per date between signal[t] and fwd_ret[t]."""
    dates = signal.index
    ic = {}
    for t in dates:
        s = signal.loc[t]
        f = fwd_ret.loc[t]
        mask = s.notna() & f.notna()
        if mask.sum() < min_valid:
            continue
        ic[t] = s[mask].rank().corr(f[mask].rank())
    return pd.Series(ic)


def ic_metrics(ic_series, signal, fwd_ret, horizons=(1, 2, 3, 5, 10, 20, 30), label=''):
    """IC, ICIR, hit rate, coverage, decay, by-year for a signal vs forward returns."""
    ic = ic_series.dropna()
    if len(ic) == 0:
        return {'error': 'no valid dates'}
    out = {
        'n_dates': int(len(ic)),
        'ic': float(ic.mean()),
        'icir': float(ic.mean() / ic.std()) if ic.std() > 0 else 0.0,
        'hit': float((ic > 0).mean()),
        'coverage': float(signal.notna().mean().mean()),
    }
    decay = {}
    for h in horizons:
        fwd = fwd_ret.shift(-h)
        ic_h = daily_ic_series(signal, fwd)
        ic_h = ic_h.dropna()
        if len(ic_h) > 0:
            decay[str(h)] = float(ic_h.mean())
    out['decay'] = decay
    yr = {}
    for y, grp in ic.groupby(ic.index.year):
        yr[str(y)] = {'ic': float(grp.mean()),
                      'icir': float(grp.mean() / grp.std()) if grp.std() > 0 else 0.0,
                      'n': int(len(grp))}
    out['by_year'] = yr
    # recent 12m / 6m
    for win, wname in [('365D', 'recent_12m'), ('182D', 'recent_6m')]:
        sub = ic[ic.index >= (ic.index.max() - pd.Timedelta(win))]
        if len(sub) >= 20:
            out[wname] = {'ic': float(sub.mean()),
                          'icir': float(sub.mean() / sub.std()) if sub.std() > 0 else 0.0,
                          'n': int(len(sub))}
    out['label'] = label
    return out


def signal_correlation_matrix(sig_new, lib_signals):
    """Pairwise Pearson rho over all (date, asset) pairs of signal artifacts; return max |rho|."""
    rows = []
    new_stacked = sig_new.stack()
    for name, s in lib_signals.items():
        other = s.stack()
        both = pd.concat([new_stacked.rename('a'), other.rename('b')], axis=1).dropna()
        if len(both) < 30:
            rho = np.nan
        else:
            rho = float(both['a'].corr(both['b']))
        rows.append((name, rho))
    maxabs = max((abs(r) for _, r in rows if not np.isnan(r)), default=0.0)
    return maxabs, rows


def format_metrics(m, top=8):
    lines = [f"[{m.get('label','')}] n_dates={m.get('n_dates')} IC={m.get('ic'):.4f} ICIR={m.get('icir'):.3f} "
             f"hit={m.get('hit'):.3f} coverage={m.get('coverage'):.3f}"]
    dec = m.get('decay', {})
    if dec:
        lines.append("  decay(ic): " + " ".join(f"{k}d={v:.4f}" for k, v in sorted(dec.items(), key=lambda x: int(x[0]))))
    for w in ('recent_12m', 'recent_6m'):
        if w in m:
            lines.append(f"  {w}: IC={m[w]['ic']:.4f} ICIR={m[w]['icir']:.3f} n={m[w]['n']}")
    yr = m.get('by_year', {})
    if yr:
        ys = sorted(yr.items())
        lines.append("  by_year: " + " ".join(f"{y}:{v['ic']:.3f}/{v['icir']:.2f}" for y, v in ys))
    return "\n".join(lines)
