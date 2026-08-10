"""miner_1 2026-07-30: quarantine redundant members of post-deprecation duplicate clusters.

Post-deprecation pairwise audit found two remaining redundancy clusters.
Deterministic rule: keep the better (by full-window ICIR / drift status) member;
quarantine the rest with a machine-readable reason (mirrors worldline_pairwise policy).

Cluster A (mean-reversion / range-position), rho >= 0.83:
  keep rsi_14d (full ICIR 0.129; recent-1y IC +0.090, ICIR 0.264)
  quarantine bollinger_z_20d (rho 0.854 vs rsi_14d), high_low_range_pos_20 (rho 0.838 vs rsi_14d)

Cluster B (equity beta), rho 0.747:
  keep spx_beta_60 (full IC +0.083, ICIR 0.199)
  quarantine btc_beta_60 (rho 0.747 vs spx_beta_60; recent-1y IC -0.020, ICIR -0.056)
"""
import json, os, glob

def quarantine(fid, reason, main_vs):
    src = f'factors/{fid}.json'
    if not os.path.exists(src):
        print(f'  {fid}: not in root; skip', flush=True)
        return
    dst = f'factors/quarantine/{fid}.json'
    d = json.load(open(src))
    art = d.get('signal_artifact')
    art_dst = None
    if art and os.path.exists(f'factors/{art}'):
        art_dst = f'factors/quarantine/{art}'
        os.replace(f'factors/{art}', art_dst)
    if os.path.exists(dst):
        os.remove(dst)
    os.replace(src, dst)
    d['validation']['status'] = 'QUARANTINED'
    d['validation']['last_validated'] = '2026-07-30'
    d['validation']['regime_notes'] = (
        d.get('validation', {}).get('regime_notes', '') +
        f' | QUARANTINED 2026-07-30: pairwise rho {d.get("validation",{}).get("metrics",{}).get("max_abs_library_correlation", 0.5):.3f} '
        f'>= 0.5 vs {main_vs}; duplicative signal, deterministic worldline_pairwise_signal_quality_v1 boundary.')
    json.dump(d, open(dst, 'w'), indent=2, default=str)
    reason_p = dst + '.reason.json'
    json.dump({'source': f'{fid}.json', 'reason': reason, 'contract': {
        'ic_threshold': 0.007, 'icir_threshold': 0.084,
        'correlation_threshold': 0.5, 'library_capacity': 30, 'active_top_k': 10}}, open(reason_p, 'w'), indent=1)
    print(f'  {fid} -> factors/quarantine/{fid}.json' + (f' (+ artifact)' if art_dst else ''), flush=True)

print('=== quarantine redundant members ===', flush=True)
quarantine('bollinger_z_20d',
           'max_abs_library_correlation 0.854 vs effective rsi_14d exceeds correlation_threshold 0.5; '
           'rsi_14d retained (higher ICIR 0.129, better recent-1y IC +0.090)',
           'rsi_14d')
quarantine('high_low_range_pos_20',
           'max_abs_library_correlation 0.838 vs effective rsi_14d exceeds correlation_threshold 0.5; '
           'rsi_14d retained (higher ICIR, better recent-1y drift)',
           'rsi_14d')
quarantine('btc_beta_60',
           'max_abs_library_correlation 0.747 vs effective spx_beta_60 exceeds correlation_threshold 0.5; '
           'spx_beta_60 retained (full-window IC +0.083 vs +0.056; btc_beta_60 recent-1y ICIR -0.056 = drift)',
           'spx_beta_60')

print('\n=== remaining effective set ===', flush=True)
for p in sorted(glob.glob('factors/*.json')):
    if 'ensemble' in p or 'deprecated' in p:
        continue
    d = json.load(open(p))
    m = d['validation']['metrics']
    print(f'  {d["factor_id"]:26s} {d["validation"]["status"]:10s} IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} '
          f'rho={m.get("max_abs_library_correlation", 0):.3f} (vs {m.get("max_corr_library_id")})', flush=True)

print('DONE')
