"""miner_2 2031-05-29: deprecate flip_mom_20x10 and mom_diff_20_60.
Revalidation as-of 2031-05-28 shows both fail:
- flip_mom_20x10: recent window IC reversed (60d IC=-0.0813 ICIR=-0.24, 180d IC=-0.0287), full-sample +0.0361 but 2024+ regime no longer clears |ICIR|>=0.084.
- mom_diff_20_60: full-sample ICIR now 0.0808 < 0.084 gate AND recent windows negative (60d IC=-0.0538).
Rename to <id>_deprecated.json and set validation.status=DEPRECATED.
"""
import json, os, glob

DEPRECATE = ['flip_mom_20x10', 'mom_diff_20_60']
ASOF = '2031-05-28'

for fid in DEPRECATE:
    src = f'factors/{fid}.json'
    dst = f'factors/{fid}_deprecated.json'
    if not os.path.exists(src):
        print(f'MISSING {src}')
        continue
    d = json.load(open(src))
    d['validation']['status'] = 'DEPRECATED'
    d['validation'].setdefault('metrics', {})
    d['validation']['deprecated_on'] = ASOF
    d['validation']['deprecation_reason'] = (
        'Revalidation 2031-05-28: short/medium window IC reversed to negative '
        '(' + fid + '). See deprecation script evidence.')
    # write to new file
    json.dump(d, open(dst, 'w'), indent=2)
    # remove original (move semantics via rename)
    os.remove(src)
    print(f'DEPRECATED {fid} -> {dst}')
    # verify reload
    rr = json.load(open(dst))
    assert rr['validation']['status'] == 'DEPRECATED'
    assert rr['factor_id'] == fid
    print(f'  verified reload: id={rr["factor_id"]} status={rr["validation"]["status"]}')

print('\nRemaining non-bak factor files:')
for f in sorted(glob.glob('factors/*.json')):
    if not f.endswith('.bak') and 'evicted' not in f and 'rejected' not in f:
        try:
            d = json.load(open(f)); print(' ', f, d.get('validation',{}).get('status'))
        except Exception as e:
            print(' ', f, 'ERR', e)