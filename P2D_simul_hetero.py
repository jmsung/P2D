
# Simulate P2D in Churchman papep (Jongmin Sung)

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize 
import random

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
    return result
    
# Parameters
mu_m = 8.6
mu_s = 0.01
mu1 = mu_m - mu_s
mu2 = mu_m + mu_s
s_m = 7.9 # sigma mean
s_s = s_m/3.0 # sigma sigma
s_t = (s_m**2 + s_s**2)**0.5
s_shape = (s_m/s_s)**2.0
s_scale = (s_s**2.0)/s_m
N = 1261
bin2 = 50
step = 0.001
mu_range = np.arange(mu_m-1, mu_m+1, step)

# Generate a dataset 
s_i = np.random.gamma(s_shape, s_scale, N)
mu1_i = np.array([mu1]*int(N/2))
mu2_i = np.array([mu2]*(N-int(N/2)))
#mu_i = np.concatenate((mu1_i, mu2_i)); random.shuffle(mu_i)
mu_i = mu_m + mu_s * np.random.randn(N)
x = s_i * np.random.randn(N) + mu_i
y = s_i * np.random.randn(N)  
r = (x**2.0 + y**2.0)**0.5

# P2D fitting with theoretical function and MLE 
result0 = P2D_MLE_mu_s(mu_m, s_m, r); mu0, s0 = result0["x"]; score0 = result0["fun"]  
result1 = P2D_MLE_mu(mu_m, s_m, r); mu_sm = result1["x"]; score1 = result1["fun"] 
result2 = P2D_MLE_mu(mu_m, s_t, r); mu_st = result2["x"]; score2 = result2["fun"]
result3 = P2D_MLE_mu(mu_m, s_i, r); mu_si = result3["x"]; score3 = result3["fun"] 

# LogLikelihood surface 
LL_sm = np.zeros(len(mu_range))
LL_st = np.zeros(len(mu_range))
LL_si = np.zeros(len(mu_range))

i = 0
for mu_scan in mu_range:
    LL_sm[i] = abs(Loglike_P2D_mu(mu_scan, s_m, r)); 
    LL_st[i] = abs(Loglike_P2D_mu(mu_scan, s_t, r));     
    LL_si[i] = abs(Loglike_P2D_mu(mu_scan, s_i, r));     
    i+=1

# Error estimation
dLL_sm = (LL_sm[1:] - LL_sm[0:-1])/step
dLL_st = (LL_st[1:] - LL_st[0:-1])/step
dLL_si = (LL_si[1:] - LL_si[0:-1])/step

Information_sm = (dLL_sm[1:] - dLL_sm[0:-1])/step
Information_st = (dLL_st[1:] - dLL_st[0:-1])/step
Information_si = (dLL_si[1:] - dLL_si[0:-1])/step

mu_std_sm = 1/(Information_sm[np.argmin(abs(mu_range-mu_sm))-1])**0.5
mu_std_st = 1/(Information_st[np.argmin(abs(mu_range-mu_st))-1])**0.5
mu_std_si = 1/(Information_si[np.argmin(abs(mu_range-mu_si))-1])**0.5

# Figure 1
plt.close('all')
fig1 = plt.figure(1)

# sp1. Mu distribution
sp1 = fig1.add_subplot(3,4,1)
sp1.hist(mu_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
title1 = 'Mu (%.1f +/- %.1f)' % (mu_m, mu_s)
sp1.set_title(title1)
sp1.axvline(x=mu_m, color='k', linewidth=0.5)

# sp2. Sigma distribution
sp2 = fig1.add_subplot(3,4,2)
sp2.hist(s_i, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
title2 = 'Sigma (%.1f +/- %.1f)' % (s_m, s_s)
sp2.set_title(title2)
sp2.axvline(x=s_m, color='k', linewidth=0.5)

# sp3. 2D localization histogram
sp3 = fig1.add_subplot(3,4,3)
sp3.hist2d(x, y, bins=bin2)
title3 = '2D Localization (N = %d)' % (N) 
sp3.set_title(title3)
sp3.set_aspect('equal')
sp3.axhline(y=0, color='w', linewidth=0.5)
sp3.axvline(x=0, color='w', linewidth=0.5)

# sp4. Histogram in X location
sp4 = fig1.add_subplot(3,4,4)
hist4 = sp4.hist(x, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp4.set_title('Localization in X')
sp4.axvline(x=0, color='k', linewidth=0.5)

# sp5. Euclidean distance - histogram
sp5 = fig1.add_subplot(3,4,5)
hist5 = sp5.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
nc = N*(hist5[1][1] - hist5[1][0])
xx = np.linspace(max(min(r), 0), max(r), 100)
sp5.set_title('Euclidean distance (R)')
sp5.axvline(x=mu_m, color='k', linewidth=0.5)

# sp6. R vs sigma 2D histogram
sp6 = fig1.add_subplot(3,4,6)
sp6.hist2d(r, s_i, bins=bin2)
sp6.set_aspect('equal')
sp6.axhline(y=s_m, color='w', linewidth=0.5)
title6 = 'R vs Sigma (corr = %.2f)' % (np.corrcoef(r, s_i)[0,1])
sp6.set_title(title6)
plt.xlabel('Euclidean distance')
plt.ylabel('Sigma')

# sp7. LogLikelihood plot
sp7 = fig1.add_subplot(3,4,7)
sp7.plot(mu_range, LL_si, 'k-')
plt.title('LogLikelihood')

# sp8. Information 
sp8 = fig1.add_subplot(3,4,8)
sp8.plot(mu_range[1:-1], Information_si, 'k-')
plt.title('Fisher Information')


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
title10 = 'P2D_sm (mu=%.1f+/-%.1f, er=%.1f%%, sc=%d)'  % (mu_sm, mu_std_sm, 100*(mu_sm/mu_m-1), score1)
sp10.set_title(title10)

# sp11. P2D MLE with mu, given st
sp11 = fig1.add_subplot(3,4,11)
sp11.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp11.plot(xx, nc*P2D(mu_st, s_t, xx), 'r', linewidth=2)
title11 = 'P2D_st (mu=%.1f+/-%.1f, er=%.1f%%, sc=%d)'  % (mu_st, mu_std_st, 100*(mu_st/mu_m-1), score2)
sp11.set_title(title11)

# sp12. P2D MLE with mu, given si
sp12 = fig1.add_subplot(3,4,12)
sp12.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
P2Ds = np.zeros((N, len(xx)), dtype=float)
for i in range(len(s_i)):
    P2Ds[i] = P2D(mu_si, s_i[i], xx)
P2Dsm = np.mean(P2Ds, axis=0)
sp12.plot(xx, nc*P2Dsm, 'r', linewidth=2)
title12 = 'P2D_si (mu=%.1f+/-%.1f, er=%.1f%%, sc=%d)'  % (mu_si, mu_std_si, 100*(mu_si/mu_m-1), score3)
sp12.set_title(title12)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.3, hspace=0.3)
plt.show()



