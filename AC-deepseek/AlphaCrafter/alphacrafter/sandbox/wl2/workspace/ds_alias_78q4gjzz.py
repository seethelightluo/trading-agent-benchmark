with open('strategy.py') as f:
    lines = f.readlines()
for i in range(120, 240):
    print(i+1, lines[i].rstrip())
