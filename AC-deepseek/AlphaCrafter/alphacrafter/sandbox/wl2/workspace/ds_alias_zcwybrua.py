lines = open('strategy.py').read().splitlines()
for i in range(379, min(530, len(lines))):
    print(f'{i+1}: {lines[i]}')