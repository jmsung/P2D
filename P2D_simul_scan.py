
# Simulate P2D in Churchman paper (Jongmin Sung)

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from scipy.optimize import minimize 
import time


pi = 3.141592

def P2D(m, s, r): 
    """P2D(r|m,s) = (r/s^2)*exp(-(m^2+r^2)/(2*s^2))*I0(r*m/s^2) Eq. (4)
    where, I0 = Modified Bessel function of order 0. """   
    P2D = (r/s**2)*np.exp(-(m**2 + r**2)/(2*s**2))*sp.i0(r*m/s**2)  
    iNaNs = np.isnan(P2D)
    P2D[iNaNs] = 1e-300
    P2D[P2D < 1e-300] = 1e-300
    return P2D
    
def P2D_approx(m, s, r): 
    P2D = (r/(2*pi*s*m))**0.5 * np.exp(-(r-m)**2/(2*s**2))
    iNaNs = np.isnan(P2D)
    P2D[iNaNs] = 1e-300
    P2D[P2D < 1e-300] = 1e-300
    return P2D 

def LL_P2D_mu_s(param, r): 
    mu, s = np.abs(param)   
    return np.sum(np.log10(P2D(mu, s, r)))

def P2D_MLE_mu_s(mu, s, r): # original P2D MLE
    fun = lambda *args: -LL_P2D_mu_s(*args)
    p0 = [0.1, 10]
    bnds = [(0.001, 2*mu), (0.001, 2*st)]
    result = minimize(fun, p0, method='SLSQP', bounds=bnds, args=r)
    return result

def LL_P2D_mu(param, s, r):
    mu = np.abs(param)    
    return np.sum(np.log10(P2D(mu, s, r))) 
    
def P2D_MLE_mu(mu, s, r): # P2D MLE with fixed mean sigma
    fun = lambda *args: -LL_P2D_mu(*args)
    p0 = [0.1]
    bnds = [(0.001, 2*mu)]
    result = minimize(fun, p0, method='SLSQP', bounds=bnds, args=(s, r)) 
    return result
    

# Parameters
mu = 10.0 
sm = 11.0 # sigma mean
ss = sm/3.0 # sigma sigma
st = (sm**2 + ss**2)**0.5
N = 1000

sm_range = np.arange(1, 100, 5)
repeat = 100

# Get a dataset 
shape = (sm/ss)**2.0
scale = (ss**2.0)/sm

si = np.random.gamma(shape, scale, N)
x = si * np.random.randn(N) + mu
y = si * np.random.randn(N)
r = (x**2.0 + y**2.0)**0.5

# P2D fitting with theoretical function and MLE 
result0 = P2D_MLE_mu_s(mu, sm, r); mu0, s0 = result0["x"]  
result1 = P2D_MLE_mu(mu, sm, r); mu_sm = result1["x"] 
result2 = P2D_MLE_mu(mu, st, r); mu_st = result2["x"]
result3 = P2D_MLE_mu(mu, si, r); mu_si = result3["x"] 

# Figure 1
plt.close('all')
fig1 = plt.figure(1)

# Sigma distribution
sp11 = fig1.add_subplot(222)
sp11.hist(si, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
title11 = 'Sigma distribution (mean = %.1f, sd = %.1f)' % (sm, ss)
sp11.set_title(title11)

# 2D location
sp12 = fig1.add_subplot(221)
sp12.plot(x, y, 'b.', markersize=2, alpha=0.2)
sp12.plot(mu, 0, 'r.', markersize = 10)
title12 = '2D Location (mu = %.1f, N = %d)' % (mu, N)
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
title21 = 'P2D MLE (mu_fit = %.1f, error = %.1f %%)' % (mu0, 100*(mu0/mu-1))
sp21.set_title(title21)

# P2D MLE with mu, given sm
sp22 = fig2.add_subplot(222)
sp22.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp22.plot(xx, nc*P2D(mu_sm, sm, xx), 'r', linewidth=2)
title22 = 'P2D MLE, fixed sigma_mean = %.1f (mu_fit = %.1f, error = %.1f %%)'  % (sm, mu_sm, 100*(mu_sm/mu-1))
sp22.set_title(title22)

# P2D MLE with mu, given st
sp23 = fig2.add_subplot(223)
sp23.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
sp23.plot(xx, nc*P2D(mu_st, st, xx), 'r', linewidth=2)
title23 = 'P2D MLE, fixed sigma_total = %.1f (mu_fit = %.1f, error = %.1f %%)'  % (st, mu_st, 100*(mu_st/mu-1))
sp23.set_title(title23)

# P2D MLE with mu, given si
sp24 = fig2.add_subplot(224)
sp24.hist(r, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
P2Ds = np.zeros((N, len(xx)), dtype=float)
for i in range(len(si)):
    P2Ds[i] = P2D(mu_si, si[i], xx)
P2Dsm = np.mean(P2Ds, axis=0)
sp24.plot(xx, nc*P2Dsm, 'r', linewidth=2)
title24 = 'P2D MLE, fixed sigma_individual (mu_fit = %.1f, error = %.1f %%)'  % (mu_si, 100*(mu_si/mu-1))
sp24.set_title(title24)



###########################################################################
# Scan over sigma to compare

mu0r = np.empty((3, len(sm_range))); 
mu1r = np.empty((3, len(sm_range)));
mu2r = np.empty((3, len(sm_range)));
mu3r = np.empty((3, len(sm_range)));

start = time.clock()
j=0
for sm in sm_range:
    mu0 = []; mu1 = []; mu2 = []; mu3 = []
    su0 = []; su1 = []; su2 = []; su3 = [];
    
    for i in range(repeat):
        ss = sm/3.0 # sigma sigma
        st = (sm**2 + ss**2)**0.5
        shape = (sm/ss)**2.0
        scale = (ss**2.0)/sm
        si = np.random.gamma(shape, scale, N)
        x = si * np.random.randn(N) + mu
        y = si * np.random.randn(N)
        r = (x**2.0 + y**2.0)**0.5

        # P2D fitting with theoretical function and MLE (Initial guess = (mu+10, s+10))
        result0 = P2D_MLE_mu_s(mu, sm, r); mu0.append(result0["x"][0]); su0.append(result0["success"])  
        result1 = P2D_MLE_mu(mu, sm, r); mu1.append(result1["x"]); su1.append(result1["success"])
        result2 = P2D_MLE_mu(mu, st, r); mu2.append(result2["x"]); su2.append(result2["success"])
        result3 = P2D_MLE_mu(mu, si, r); mu3.append(result3["x"]); su3.append(result3["success"])        
        
    mu0r[:,j] = [np.nanmean(mu0), np.nanstd(mu0), sum(su0)/repeat*100]
    mu1r[:,j] = [np.nanmean(mu1), np.nanstd(mu1), sum(su1)/repeat*100]
    mu2r[:,j] = [np.nanmean(mu2), np.nanstd(mu2), sum(su2)/repeat*100]
    mu3r[:,j] = [np.nanmean(mu3), np.nanstd(mu3), sum(su3)/repeat*100]
    j+=1
    
    done = sm/max(sm_range)*100
    now = time.clock()
    spent = (now-start)/60 # time passed in min
    print(spent*(100-done)/done)
  
    
fig3 = plt.figure(3)
sp31 = fig3.add_subplot(241); 
sp31.errorbar(x=sm_range, y=mu0r[0], yerr=mu0r[1], color='k'); 
sp31.axhline(y=mu, color='k', linewidth=0.5)
sp31.axis([min(sm_range)-1, max(sm_range)+1, 0, 20])
sp31.set_title('mu=%.1f (init=1), repeat=%d, particle=%d  \nP2D MLE' % (mu, repeat, N))
sp31.set_ylabel('Estimation of mu')

sp32 = fig3.add_subplot(242); 
sp32.errorbar(x=sm_range, y=mu1r[0], yerr=mu1r[1], color='g'); 
sp32.axhline(y=mu, color='k', linewidth=0.5)
sp32.axis([min(sm_range)-1, max(sm_range)+1, 0, 20])
sp32.set_title('P2D MLE, fixed sigma_mean')

sp33 = fig3.add_subplot(243); 
sp33.errorbar(x=sm_range, y=mu2r[0], yerr=mu2r[1], color='b'); 
sp33.axhline(y=mu, color='k', linewidth=0.5)
sp33.axis([min(sm_range)-1, max(sm_range)+1, 0, 20])
sp33.set_title('P2D MLE, fixed sigma_total')

sp34 = fig3.add_subplot(244); 
sp34.errorbar(x=sm_range, y=mu3r[0], yerr=mu3r[1], color='r'); 
sp34.axhline(y=mu, color='k', linewidth=0.5)
sp34.axis([min(sm_range)-1, max(sm_range)+1, 0, 20])
sp34.set_title('P2D MLE, fixed sigma_individual')

sp35 = fig3.add_subplot(245); 
sp35.plot(sm_range, mu0r[2], 'ko')
sp35.axhline(y=100, color='k', linewidth=0.5)
sp35.axis([min(sm_range)-1, max(sm_range)+1, 0, 110])
sp35.set_ylabel('% Success')
sp35.set_xlabel('Sigma_mean')

sp36 = fig3.add_subplot(246); 
sp36.plot(sm_range, mu1r[2], 'go')
sp36.axhline(y=100, color='k', linewidth=0.5)
sp36.axis([min(sm_range)-1, max(sm_range)+1, 0, 110])
sp36.set_xlabel('Sigma_mean')

sp37 = fig3.add_subplot(247); 
sp37.plot(sm_range, mu2r[2], 'bo')
sp37.axhline(y=100, color='k', linewidth=0.5)
sp37.axis([min(sm_range)-1, max(sm_range)+1, 0, 110])
sp37.set_xlabel('Sigma_mean')

sp38 = fig3.add_subplot(248); 
sp38.plot(sm_range, mu3r[2], 'ro')
sp38.axhline(y=100, color='k', linewidth=0.5)
sp38.axis([min(sm_range)-1, max(sm_range)+1, 0, 110])
sp38.set_xlabel('Sigma_mean')



plt.show()


"""
To-do
- sm = 1-21-1, m=1000, N=100/1000/10000
- change ss
- MLE plot 
"""
      
                  


