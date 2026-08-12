lines = open('memory.txt').read().strip().split('\n')
for i, l in enumerate(lines[-5:]):
    print(f"{i:3d}| {l}")