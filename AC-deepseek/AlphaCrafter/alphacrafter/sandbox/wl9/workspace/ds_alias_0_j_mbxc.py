with open('scripts/miner_2_20330721_revalidate_sweep.py') as f:
    lines = f.readlines()
print('total lines:', len(lines))
for i, line in enumerate(lines):
    if 'report("' in line:
        print(i, line.strip())