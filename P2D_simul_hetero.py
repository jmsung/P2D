
# Simulate P2D in Churchman papep (Jongmin Sung)

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize 
import time

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
    p0 = [mu_m, s_t]
    bnds = [(0.001, 2*mu_m), (0.001, 2*s_t)]
    result = minimize(fun, p0, method='SLSQP', bounds=bnds, args=r)
#    print(result["x"]); print(result["success"])
    return result

def Loglike_P2D_mu(param, s, r):
    mu = np.abs(param)      
    return np.sum(np.log10(P2D(mu, s, r)))

def P2D_MLE_mu(mu, s, r): # P2D MLE with fixed mean sigma
    fun = lambda *args: -Loglike_P2D_mu(*args)
#    p0 = [10*mu*np.random.rand()]
    p0 = [mu_m]
    bnds = [(0.001, 2*mu_m)]
    result = minimize(fun, p0, method='SLSQP', bounds=bnds, args=(s, r)) 
#    print(result["x"]); print(result["success"])
    return result
    

# Parameters
mu_m = 8.0 
mu_s = 4.0
s_m = 8.0 # sigma mean
s_s = s_m/3.0 # sigma sigma
s_t = (s_m**2 + s_s**2)**0.5
N = 10000

# Get a dataset 
shape = (s_m/s_s)**2.0
scale = (s_s**2.0)/s_m

s_i = np.random.gamma(shape, scale, N)
mu_i = mu_m + mu_s * np.random.randn(N)
x = s_i * np.random.randn(N) + mu_i
y = s_i * np.random.randn(N)  
r = (x**2.0 + y**2.0)**0.5

# P2D fitting with theoretical function and MLE 
result0 = P2D_MLE_mu_s(mu_m, s_m, r); mu0, s0 = result0["x"]  
result1 = P2D_MLE_mu(mu_m, s_m, r); mu_sm = result1["x"] 
result2 = P2D_MLE_mu(mu_m, s_t, r); mu_st = result2["x"]
result3 = P2D_MLE_mu(mu_m, s_i, r); mu_si = result3["x"] 

# Figure 1
plt.close('all')
fig1 = plt.figure(1)

# Mu distribution
sp11 = fig1.add_subplot(222)
sp11.hist(mu

# Sigma distribution
sp11 = fig1.add_subplot(222)
sp11.hist(mu_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
title11 = 'Sigma distribution (mean = %.1f, sd = %.1f)' % (s_m, s_s)
sp11.set_title(title11)

# 2D location
sp12 = fig1.add_subplot(221)
sp12.plot(x, y, 'b.', markersize=2, alpha=0.2)
sp12.plot(mu_m, 0, 'r.', markersize = 10)
title12 = '2D Location (mu = %.1f, N = %d)' % (mu_m, N)
sp12.set_title(title12)
sp12.set_aspect('equal', 'datalim')
sp12.axhline(y=0, color='k', linewidth=0.5)
sp12.axvline(x=0, color='k', linewidth=0.5)

# Projection of location in X
sp14 = fig1.add_subplot(223)
hist14 = sp14.hist(x, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp14.set_title('Location in X')
sp14.axvline(x=0, color='k', linewidth=0.5)

# Euclidean distance - histogram
sp13 = fig1.add_subplot(224)
hist13 = sp13.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
nc = N*(hist13[1][1] - hist13[1][0])
xx = np.linspace(max(min(r), 0), max(r), 100)
sp13.set_title('Euclidean distance')

# Figure 2
fig2 = plt.figure(2)

# P2D MLE with mu, sigma
sp21 = fig2.add_subplot(221)
sp21.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp21.plot(xx, nc*P2D(mu0, s0, xx), 'r', linewidth=2)
title21 = 'P2D MLE (mu_fit = %.1f, error = %.1f %%)' % (mu0, 100*(mu0/mu_m-1))
sp21.set_title(title21)


# P2D MLE with mu, given st
sp23 = fig2.add_subplot(223)
sp23.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp23.plot(xx, nc*P2D(mu_st, s_t, xx), 'r', linewidth=2)
title23 = 'P2D MLE, fixed sigma_total = %.1f (mu_fit = %.1f, error = %.1f %%)'  % (s_t, mu_st, 100*(mu_st/mu_m-1))
sp23.set_title(title23)

# P2D MLE with mu, given si
sp24 = fig2.add_subplot(224)
sp24.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
P2Ds = np.zeros((N, len(xx)), dtype=float)
for i in range(len(s_i)):
    P2Ds[i] = P2D(mu_si, s_i[i], xx)
P2Dsm = np.mean(P2Ds, axis=0)
sp24.plot(xx, nc*P2Dsm, 'r', linewidth=2)
title24 = 'P2D MLE, fixed sigma_individual (mu_fit = %.1f, error = %.1f %%)'  % (mu_si, 100*(mu_si/mu_m-1))
sp24.set_title(title24)

plt.show()



