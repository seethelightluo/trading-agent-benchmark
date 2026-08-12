lines = open('memory.txt').read().strip().split('\n')
print("total lines:", len(lines))
for i, l in enumerate(lines[-18:]):
    print(f"{i:3d}| {l[:300]}")