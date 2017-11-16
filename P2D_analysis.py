# -*- coding: utf-8 -*-
# Simulate P2D in Churchman papep (Jongmin Sung, 6/9/17)

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

# Sigma distribution
sp11 = fig1.add_subplot(231)
sp11.hist(si, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
title11 = 'Sigma distribution (mean = %.1f, sd = %.1f)' % (np.mean(si), np.std(si))
sp11.set_title(title11)

# Euclidean distance - histogram
sp12 = fig1.add_subplot(232)
hist12 = sp12.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
nc = N*(hist12[1][1] - hist12[1][0])
xx = np.linspace(max(min(r), 0), max(r), 100)
title12 = 'Euclidean distance (N = %d)' % (N)
sp12.set_title(title12)

# P2D MLE with mu, sigma
sp13 = fig1.add_subplot(233)
sp13.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp13.plot(xx, nc*P2D(mu0, s0, xx), 'r', linewidth=2)
title13 = 'P2D MLE (mu_fit = %.1f, s_fit = %.1f)' % (mu0, s0)
sp13.set_title(title13)

# P2D MLE with mu, given sm
sp14 = fig1.add_subplot(234)
sp14.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp14.plot(xx, nc*P2D(mu_sm, sm, xx), 'r', linewidth=2)
title14 = 'P2D MLE, fixed sigma_mean (mu_fit = %.1f)'  % (mu_sm)
sp14.set_title(title14)

# P2D MLE with mu, given st
sp15 = fig1.add_subplot(235)
sp15.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp15.plot(xx, nc*P2D(mu_st, st, xx), 'r', linewidth=2)
title15 = 'P2D MLE, fixed sigma_total (mu_fit = %.1f)'  % (mu_st)
sp15.set_title(title15)

# P2D MLE with mu, given si
sp16 = fig1.add_subplot(236)
sp16.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
P2Ds = np.zeros((N, len(xx)), dtype=float)
for i in range(len(si)):
    P2Ds[i] = P2D(mu_si, si[i], xx)
P2Dsm = np.mean(P2Ds, axis=0)
sp16.plot(xx, nc*P2Dsm, 'r', linewidth=2)
title16 = 'P2D MLE, fixed sigma_individual (mu_fit = %.1f)'  % (mu_si)
sp16.set_title(title16)

plt.show()


