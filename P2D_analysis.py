# -*- coding: utf-8 -*-
# Simulate P2D in Churchman papep (Jongmin Sung)

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize 


# P2D(r|mu,s) = (r/s^2)*exp(-(mu^2+r^2)/(2*s^2))*I0(r*mu/s^2) Eq. (4)
# I0 = Modified Bessel function of order 0.

def P2D(mu, s, r):   
    return (r/s**2)*np.exp(-(mu**2 + r**2)/(2*s**2))*sp.i0(r*mu/s**2)

def Loglike_P2D_mu_s(param, r): 
    mu, s = np.abs(param)   
    return np.sum(np.log10(P2D(mu, s, r)))

def P2D_MLE_mu_s(mu, s, r): # original P2D MLE
    fun = lambda *args: -Loglike_P2D_mu_s(*args)
#    p0 = [10*mu*np.random.rand(), 10*st*np.random.rand()]
    p0 = [8, 3]
    bnds = [(1, 20), (1, 20)]
    result = minimize(fun, p0, method='SLSQP', bounds=bnds, args=r)
    print(result); print('\n')
    return result

def Loglike_P2D_mu(param, s, r):
    mu = np.abs(param)      
    return np.sum(np.log10(P2D(mu, s, r)))

def P2D_MLE_mu(mu, s, r): # P2D MLE with fixed mean sigma
    fun = lambda *args: -Loglike_P2D_mu(*args)
#    p0 = [10*mu*np.random.rand()]
    p0 = [8]
    bnds = [(1, 20)]
    result = minimize(fun, p0, method='SLSQP', bounds=bnds, args=(s, r)) 
    print(result); print('\n')
    return result
    
# Open file
file_r = open("Kinesin_r.txt", "r") 
file_s = open("Kinesin_s.txt", "r") 

rl = []
sl = []

for line in file_r:
    rl.append(float(line))

for line in file_s:
    sl.append(float(line))

file_r.close()
file_s.close()

r = np.array(rl)
si = np.array(sl)
si = si*2.5


# Parameters
N = len(r)
mu = 8.0
sm = np.mean(si)
ss = np.std(si)
st = (sm**2.0 + ss**2.0)**0.5

# P2D fitting with theoretical function and MLE 
result0 = P2D_MLE_mu_s(mu, sm, r); mu0, s0 = result0["x"]  
result1 = P2D_MLE_mu(mu, sm, r); mu_sm = result1["x"] 
result2 = P2D_MLE_mu(mu, st, r); mu_st = result2["x"]
result3 = P2D_MLE_mu(mu, si, r); mu_si = result3["x"] 

# Figure 1
plt.close('all')
fig1 = plt.figure(1)

# sp1. Sigma distribution
sp1 = fig1.add_subplot(241)
sp1.hist(si, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
title1 = 'Sigma distribution (mean = %.1f, sd = %.1f)' % (np.mean(si), np.std(si))
sp1.set_title(title1)

# sp2. Euclidean distance - histogram
sp2 = fig1.add_subplot(242)
hist2 = sp2.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
nc = N*(hist2[1][1] - hist2[1][0])
xx = np.linspace(max(min(r), 0), max(r), 100)
title2 = 'Euclidean distance (N = %d)' % (N)
sp2.set_title(title2)

# sp3. R vs sigma scatter
sp3 = fig1.add_subplot(243)
sp3.plot(r, si, 'k.', markersize=5, alpha=0.5)
title3 = 'R vs Sigma (corr = %.2f)' % (np.corrcoef(r, si)[0,1]) 
sp3.set_title(title3)
sp3.set_aspect('equal')

# sp4. R vs sigma 2D histogram
sp4 = fig1.add_subplot(244)
sp4.hist2d(r, si, bins=20)
sp4.set_aspect('equal')
title4 = 'R vs sigma histogram' 
sp4.set_title(title4)

# sp5. P2D MLE with mu, sigma
sp5 = fig1.add_subplot(245)
sp5.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp5.plot(xx, nc*P2D(mu0, s0, xx), 'r', linewidth=2)
title5 = 'P2D (mu_fit = %.1f, s_fit = %.1f)' % (mu0, s0)
sp5.set_title(title5)

# sp6. P2D MLE with mu, given sm
sp6 = fig1.add_subplot(246)
sp6.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp6.plot(xx, nc*P2D(mu_sm, sm, xx), 'r', linewidth=2)
title6 = 'P2D_sm (mu_fit = %.1f)'  % (mu_sm)
sp6.set_title(title6)

# sp7. P2D MLE with mu, given st
sp7 = fig1.add_subplot(247)
sp7.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp7.plot(xx, nc*P2D(mu_st, st, xx), 'r', linewidth=2)
title7 = 'P2D_st (mu_fit = %.1f)'  % (mu_st)
sp7.set_title(title7)

# sp8. P2D MLE with mu, given si
sp8 = fig1.add_subplot(248)
sp8.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
P2Ds = np.zeros((N, len(xx)), dtype=float)
for i in range(len(si)):
    P2Ds[i] = P2D(mu_si, si[i], xx)
P2Dsm = np.mean(P2Ds, axis=0)
sp8.plot(xx, nc*P2Dsm, 'r', linewidth=2)
title8 = 'P2D_si (mu_fit = %.1f)'  % (mu_si)
sp8.set_title(title8)

plt.show()


