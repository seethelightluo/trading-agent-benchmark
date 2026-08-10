import json, glob, os

files = sorted(glob.glob('factors/*.json'))
files = [f for f in files if not f.endswith('.bak') and '.npy' not in f]

by_id = {}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    fid = d.get('factor_id')
    if fid is None:
        continue
    base = os.path.basename(f)
    has_ts = ('20260810' in base or '20260811' in base)
    if fid not in by_id or (not has_ts and by_id[fid][1]):
        by_id[fid] = (d, has_ts)

rows = []
for fid, (d, _) in by_id.items():
    v = d.get('validation', {}).get('metrics', {})
    ba = d.get('benchmark_admission', {}).get('selected_metrics', {})
    rows.append({
        'fid': fid,
        'ic1': v.get('ic1'), 'icir1': v.get('icir1'), 'hit1': v.get('hit1'),
        'ic5': v.get('ic5'), 'icir5': v.get('icir5'),
        'turn10': v.get('turnover_10d'), 'cov': v.get('coverage'),
        'adm_ic': ba.get('ic'), 'adm_icir': ba.get('icir'),
        'n_dates': v.get('n_dates'),
    })

rows.sort(key=lambda r: -(abs(r['icir1'] or 0)))
print(f"{'factor_id':<36} {'ic1':>6} {'icir1':>7} {'hit1':>6} {'ic5':>6} {'icir5':>7} {'turn10':>7} {'cov':>5} {'admIC':>6} {'admICIR':>7}")
for r in rows:
    print(f"{r['fid']:<36} {r['ic1']:>6.4f} {r['icir1']:>7.3f} {r['hit1']:>6.3f} "
          f"{r['ic5']:>6.4f} {r['icir5']:>7.3f} {r['turn10']:>7.3f} {r['cov']:>5.2f} "
          f"{r['adm_ic']:>6.4f} {r['adm_icir']:>7.3f}")

# direction/sign info: check signal_artifact or calculation for direction hint
print("\n--- direction hints (signal_artifact keys) ---")
for fid, (d, _) in sorted(by_id.items()):
    sa = d.get('signal_artifact', {})
    print(fid, "|", json.dumps(sa)[:200])
