
with open('strategy.py') as f:
    lines = f.readlines()
for i in range(570, len(lines)):
    print(f'{i+1}: {lines[i]}', end='')
print("===== HEAD =====")
for i in range(0, 40):
    print(f'{i+1}: {lines[i]}', end='')
