import subprocess, glob, re, os, sys

scripts = sorted(glob.glob('scripts/miner3_*.py'))
skip = {'miner3_20260831_collect_gates.py'}
print(f"Total miner3 scripts: {len(scripts)}")
for s in scripts:
    if os.path.basename(s) in skip:
        continue
    try:
        r = subprocess.run(['python', s], capture_output=True, text=True, timeout=280)
        out = r.stdout + r.stderr
    except Exception as e:
        print(f"\n### {s}\n  ERROR: {e}")
        continue
    # extract factor blocks with GATE lines
    blocks = re.split(r'^=== ', out, flags=re.M)
    passes = []
    results = []
    for b in blocks:
        lines = b.splitlines()
        fac = lines[0].strip() if lines else ''
        gate = [l for l in lines if 'GATE' in l]
        icline = [l for l in lines if l.startswith('  IC (h=10)')]
        if gate:
            res = gate[0].strip()
            results.append((fac, icline[0].strip() if icline else '', res))
            if 'PASS' in res:
                passes.append((fac, icline[0].strip() if icline else '', res))
    print(f"### {os.path.basename(s)}  -> {len(results)} factors, {len(passes)} PASS")
    for fac, ic, g in passes:
        print(f"   PASS: {fac} | {ic} | {g}")