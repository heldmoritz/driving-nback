import numpy as np

start = 0.011
a = 1.1
b = 0.015

def get_noise(s):
    p = np.random.rand(1)[0]
    noise = s *np.log((1.0 - p)/p)
    return noise

def tn(n):
    if n == 0:
        tick = b*5*start
    else:
        tick = n*a
        tick += getNoise(b * tn(n-1))

    return tick

e1 = tn(0)
e2 = tn(20)
t0 = start + e1
t20 = a * tn(19) + e2
print(t20)

for i in range(0, 40):
    print(tn(120))
