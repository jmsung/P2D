
# Simulate P2D in Churchman papep (Jongmin Sung)

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize 
import random
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
mu_m = 8.6
mu_s = 0.01
mu1 = mu_m - mu_s
mu2 = mu_m + mu_s
s_m = 17.9 # sigma mean
s_s = s_m/3.0 # sigma sigma
s_t = (s_m**2 + s_s**2)**0.5
s_shape = (s_m/s_s)**2.0
s_scale = (s_s**2.0)/s_m
N = 12610
bin2 = 20

# Generate a dataset 
s_i = np.random.gamma(s_shape, s_scale, N)
mu1_i = np.array([mu1]*int(N/2))
mu2_i = np.array([mu2]*(N-int(N/2)))
mu_i = np.concatenate((mu1_i, mu2_i)); random.shuffle(mu_i)
#mu_i = mu_m + mu_s * np.random.randn(N)
x = s_i * np.random.randn(N) + mu_i
y = s_i * np.random.randn(N)  
r = (x**2.0 + y**2.0)**0.5

# P2D fitting with theoretical function and MLE 
result0 = P2D_MLE_mu_s(mu_m, s_m, r); mu0, s0 = result0["x"]; score0 = result0["fun"]  
result1 = P2D_MLE_mu(mu_m, s_m, r); mu_sm = result1["x"]; score1 = result1["fun"] 
result2 = P2D_MLE_mu(mu_m, s_t, r); mu_st = result2["x"]; score2 = result2["fun"]
result3 = P2D_MLE_mu(mu_m, s_i, r); mu_si = result3["x"]; score3 = result3["fun"] 

# Figure 1
plt.close('all')
fig1 = plt.figure(1)

# sp1. Mu distribution
sp1 = fig1.add_subplot(3,4,1)
sp1.hist(mu_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
title1 = 'Mu distribution (%.1f +/- %.1f)' % (mu_m, mu_s)
sp1.set_title(title1)
sp1.axvline(x=mu_m, color='k', linewidth=0.5)

# sp2. Sigma distribution
sp2 = fig1.add_subplot(3,4,2)
sp2.hist(s_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
title2 = 'Sigma distribution (%.1f +/- %.1f)' % (s_m, s_s)
sp2.set_title(title2)
sp2.axvline(x=s_m, color='k', linewidth=0.5)

# sp3. 2D localization scatter
sp3 = fig1.add_subplot(3,4,3)
sp3.plot(x, y, 'k.', markersize=2, alpha=0.2)
sp3.plot(mu_m, 0, 'r.', markersize = 10)
title3 = '2D Locatization (mu = %.1f, N = %d)' % (mu_m, N)
sp3.set_title(title3)
sp3.set_aspect('equal')
sp3.axhline(y=0, color='k', linewidth=0.5)
sp3.axvline(x=0, color='k', linewidth=0.5)

# sp4. 2D localization histogram
sp4 = fig1.add_subplot(3,4,4)
sp4.hist2d(x, y, bins=bin2)
title4 = '2D Locatization histogram' 
sp4.set_title(title4)
sp4.set_aspect('equal')
sp4.axhline(y=0, color='w', linewidth=0.5)
sp4.axvline(x=0, color='w', linewidth=0.5)

# sp5. Histogram in X location
sp5 = fig1.add_subplot(3,4,5)
hist5 = sp5.hist(x, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp5.set_title('Location in X')
sp5.axvline(x=0, color='k', linewidth=0.5)

# sp6. Euclidean distance - histogram
sp6 = fig1.add_subplot(3,4,6)
hist6 = sp6.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
nc = N*(hist6[1][1] - hist6[1][0])
xx = np.linspace(max(min(r), 0), max(r), 100)
sp6.set_title('Euclidean distance (R)')
sp6.axvline(x=mu_m, color='k', linewidth=0.5)

# sp7. R vs sigma scatter
sp7 = fig1.add_subplot(3,4,7)
sp7.plot(r, s_i, 'k.', markersize=5, alpha=0.5)
sp7.plot(mu_m, s_m, 'r.', markersize = 10)
title7 = 'R vs Sigma (corr = %.2f)' % (np.corrcoef(r, s_i)[0,1])
sp7.set_title(title7)
sp7.set_aspect('equal')
#sp7.set_aspect('equal', 'datalim')

# sp8. R vs sigma 2D histogram
sp8 = fig1.add_subplot(3,4,8)
sp8.hist2d(r, s_i, bins=bin2)
sp8.set_aspect('equal')
sp8.axhline(y=s_m, color='w', linewidth=0.5)
sp8.axvline(x=mu_m, color='w', linewidth=0.5)
title8 = 'R vs sigma histogram' 
sp8.set_title(title8)

# sp9. P2D MLE with mu, sigma
sp9 = fig1.add_subplot(3,4,9)
sp9.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp9.plot(xx, nc*P2D(mu0, s0, xx), 'r', linewidth=2)
title9 = 'P2D (mu=%.1f, err=%.1f%%, score=%d)' % (mu0, 100*(mu0/mu_m-1), score0)
sp9.set_title(title9)

# sp10. P2D MLE with mu, given sm
sp10 = fig1.add_subplot(3,4,10)
sp10.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp10.plot(xx, nc*P2D(mu_sm, s_t, xx), 'r', linewidth=2)
title10 = 'P2D_sm (mu=%.1f, err=%.1f%%, score=%d)'  % (mu_sm, 100*(mu_sm/mu_m-1), score1)
sp10.set_title(title10)

# sp11. P2D MLE with mu, given st
sp11 = fig1.add_subplot(3,4,11)
sp11.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp11.plot(xx, nc*P2D(mu_st, s_t, xx), 'r', linewidth=2)
title11 = 'P2D_st (mu=%.1f, err=%.1f%%, score=%d)'  % (mu_st, 100*(mu_st/mu_m-1), score2)
sp11.set_title(title11)

# sp12. P2D MLE with mu, given si
sp12 = fig1.add_subplot(3,4,12)
sp12.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
P2Ds = np.zeros((N, len(xx)), dtype=float)
for i in range(len(s_i)):
    P2Ds[i] = P2D(mu_si, s_i[i], xx)
P2Dsm = np.mean(P2Ds, axis=0)
sp12.plot(xx, nc*P2Dsm, 'r', linewidth=2)
title12 = 'P2D_si (mu=%.1f, err=%.1f%%, score=%d)'  % (mu_si, 100*(mu_si/mu_m-1), score3)
sp12.set_title(title12)

plt.show()



